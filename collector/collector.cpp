#define _GNU_SOURCE 1

#include <stdlib.h>
#include <iostream>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <unordered_map>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <sys/ioctl.h>
#include <time.h>
#include <errno.h>
#include <fstream>
#include <algorithm>
#include <pthread.h>
#include <time.h>


extern "C"
{
#include "xed/xed-interface.h"
}
//#include "xed/xed-decoded-inst-api.h"
//#include "xed/xed-decode.h"

#include "elf_parser.hpp"
#include <inttypes.h>

#define RING_BUFFER_PAGES 1
#define MEM_SIGNAL (SIGRTMIN+7)
#define WP_W_SIGNAL (SIGRTMIN+3)
#define WP_R_SIGNAL (SIGRTMIN+4)
#define WP_SIGNAL (SIGRTMIN+5)
#define MAX_WATCHPOINT_WAIT 100
#define MAX_BYTES_TO_DECODE (15)

#define gettid() syscall(__NR_gettid)

xed_state_t xedState;

int data_list[10000];


int perf_event_open(struct perf_event_attr *attr,pid_t pid,int cpu,int group_fd,unsigned long flags)
{
    return syscall(__NR_perf_event_open,attr,pid,cpu,group_fd,flags);
}


struct perf_my_sample
{
    struct perf_event_header header;
                uint64_t ip;
                uint64_t addr;
};

//for memory_access perf_event
int fd;
void* rbuf;
uint64_t next_offset=0;
//for watchpoint perf_event

pid_t pid_target;  
uint64_t access_cnt=0;
int cnt_handler=0;
bool inside_tool=false;
bool target_exit=false;
clock_t last_sample_t=0;
std::unordered_map<uint64_t,uint64_t> pc_list;
bool first_store_arrive=false;
bool first_load_arrive=false;
enum DETECTOR_TARGET{
        DEADSTORE=1,
        SILENTSTORE,
        SILENTLOAD
} detector;
/*enum WP_STATE{
        wp_wait=1,
        wp_finish,
        wp_timeout
} new_wp_state;*/
//bool wp_on;
enum WP_STATE{
        WP_RUN=1,
        WP_FINISH,
        WP_WAIT_TO_CLOSE,
        WP_CLEARED
} wp_state;
enum CREATE_RET{
        CREATE_SUCCESS=0,
        CREATE_FILE_ERR,
        CREATE_BUFFER_ERR,
        OTHER_ERR
};
int wp_tid;
pthread_mutex_t wp_mutex;
pthread_mutex_t cnt_mutex;
pthread_t th1,th2;
sigset_t receive_set,receive_set_mem,receive_set_wp; //信号集


struct Sample_data{
        uint64_t addr;
        uint64_t ip;
        void* wp_r_buffer;
        int wp_r_fd;
        uint64_t wp_r_next_offset;
        uint64_t pre_ip;
        uint64_t target_addr;
}sample_data;
uint64_t write_amount;
uint64_t dead_amount;
uint64_t sample_memory_access_cnt;
std::vector<elf_parser::symbol_t> syms; 
uint8_t* program_phy_base;
uint64_t program_fini,program_init;
uint8_t* program_base;
std::unordered_map<uint64_t,int> funcLabelList;
std::ofstream memoryFile;
bool phase_get_memory=true;
enum INST_TYPE{
        store_inst=1,
        load_inst,
        unknow_inst
};


uint64_t get_func_ip(uint64_t ip){
        uint64_t func_ip=-1;
  for (int i=0;i<syms.size();i++) {
                        if(ip<syms[i].symbol_value){
                                if(i==0){func_ip=0;break;}
                                func_ip=syms[i-1].symbol_value;
                                break;
                        }
        }
        //printf("func_ip:%d %d\n",func_ip,ip);
        return func_ip;
}

bool is_target_func(uint64_t ip){
        if((ip<program_init)||(ip>program_fini)){
                return false;
        }

        //return true; //true:no filter,all pass
  
        uint64_t func_ip = get_func_ip(ip);
        std::unordered_map<uint64_t,int>::iterator it = funcLabelList.find(func_ip);
        if(it == funcLabelList.end()){
                printf("func not found: %llx (ip=%llx)\n",func_ip,ip);
                return true; //nolabel == label1
        }
        if(it->second==1){
                return true;
        }
        return false;
}

static inline void rmb(void) {
  asm volatile("lfence":::"memory");
}

int readBuffer(void *mbuf, void *buf, size_t sz) { //读取一个sample的信息
        //printf("read buffer: ");
    struct perf_event_mmap_page *hdr = (struct perf_event_mmap_page *)mbuf;
        //printf("%llx %llx ",mbuf,hdr);
        //std::cout<<hdr->data_tail<<" "<<hdr->data_head<<std::endl;
    void *data;
        size_t pgsz=sysconf(_SC_PAGESIZE);
    unsigned long tail;
    size_t avail_sz, m, c;
    size_t pgmsk = pgsz - 1;
    data = (uint8_t*)hdr + pgsz;
    tail = hdr->data_tail & pgmsk;
    avail_sz = hdr->data_head - hdr->data_tail; //Sample data not yet read, possibly more than one sample
    if (sz > avail_sz) {
        printf("\n sz > avail_sz: sz = %lu, avail_sz = %lu, hdr->data_tail = %llx, hdr->data_head = %llx\n", sz, avail_sz,hdr->data_tail,hdr->data_head);
        rmb();
        return -1;
    }
    rmb();
    c = pgmsk + 1 -  tail;
    m = c < sz ? c : sz; //Is the buffer boundary exceeded
    //std::cout<<hdr->data_tail<<" "<<hdr->data_head<<" "<<avail_sz<<" "<<tail<<" "<<m<<" "<<c<<std::endl;
        memcpy(buf, data + tail, m);
    if (sz > m)
        memcpy((uint8_t*)buf + m, data, sz - m);
    hdr->data_tail += sz;
        //printf("read buffer finish\n");
        //std::cout<<"buf content:"<<*((uint64_t*)buf)<<std::endl;
    return 0;
}

void close_bp(){ //关闭debug_register
        //printf("close breakpoint\n");
    ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_DISABLE,0);
    munmap(sample_data.wp_r_buffer,(1+RING_BUFFER_PAGES)*4096);
    close(sample_data.wp_r_fd);
        sample_data.wp_r_fd=-1;
        wp_state=WP_CLEARED;
}


CREATE_RET create_bp(uint64_t addr){ //创建新的debug_register监控
        //printf("create breakpoint for %lx \n",addr);
        wp_state=WP_RUN;

        struct perf_event_attr attr_wp;
        memset(&attr_wp,0,sizeof(struct perf_event_attr));
        attr_wp.size=sizeof(struct perf_event_attr);
        attr_wp.disabled=1;
        attr_wp.type=PERF_TYPE_BREAKPOINT; //PERF_TYPE_BREAKPOINT; //PERF_TYPE_SOFTWARE; //监控类别，breakpoint
        //attr_wp.config=0;
        //attr_wp.bp_type=HW_BREAKPOINT_RW;
        //attr_wp.bp_type=HW_BREAKPOINT_R;
        if(detector==DEADSTORE){
                attr_wp.bp_type=HW_BREAKPOINT_RW;
        }else if(detector==SILENTSTORE){
                attr_wp.bp_type=HW_BREAKPOINT_R;
        }else if(detector==SILENTLOAD){
                attr_wp.bp_type=HW_BREAKPOINT_W;
        }
        attr_wp.bp_addr=addr; 
        attr_wp.bp_len=HW_BREAKPOINT_LEN_4;
        attr_wp.sample_period=1;
        attr_wp.sample_type=PERF_SAMPLE_IP|PERF_SAMPLE_ADDR; //PERF_SAMPLE_IP;
        attr_wp.precise_ip=2;
        attr_wp.exclude_user=0;
        attr_wp.exclude_kernel=1;
        attr_wp.exclude_hv=1;

        //read wp
    sample_data.wp_r_fd=perf_event_open(&attr_wp,pid_target,-1,-1,0); // open perf_breakpoint fd
    if(sample_data.wp_r_fd<0){
        //perror("breakpoint: perf_event_open() failed!");
                printf("breakpoint: perf_event_open() failed! pid_target=%d\n",pid_target);
                wp_state=WP_CLEARED;
        return CREATE_FILE_ERR;
    }else{
                //printf("wp_r_fd = %d ",sample_data.wp_r_fd);
        }
        sample_data.wp_r_buffer=mmap(0,(1+RING_BUFFER_PAGES)*4096,PROT_READ|PROT_WRITE,MAP_SHARED,sample_data.wp_r_fd,0); //分配记录用的buffer
        sample_data.wp_r_next_offset=0;
    if(sample_data.wp_r_buffer==0){
        perror("break_point: mmap() failed!");
                wp_state=WP_CLEARED;
                close(sample_data.wp_r_fd);
                sample_data.wp_r_fd=-1;
        return CREATE_BUFFER_ERR;
    }else{
                //printf("new watchpoint buffer: %llx\n",sample_data.wp_r_buffer);
        }
        sample_data.target_addr=attr_wp.bp_addr;

        first_store_arrive=false;
        first_load_arrive=false;

        fcntl(sample_data.wp_r_fd,F_SETFL,O_RDWR|O_NONBLOCK|O_ASYNC);
        //fcntl(wp_fd,F_SETFL,O_RDWR|O_ASYNC);
    fcntl(sample_data.wp_r_fd,F_SETSIG,WP_SIGNAL);
        //fcntl(sample_data.wp_r_fd,F_SETOWN,getpid());
        fcntl(sample_data.wp_r_fd,F_SETOWN,getpid());
        ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_RESET,0);
        ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);

        //printf("create breakpoint finish\n",addr);
        return CREATE_SUCCESS;
}

static inline void XedInit()
{
        xed_tables_init();
        xed_state_init (&xedState, XED_MACHINE_MODE_LONG_64, XED_ADDRESS_WIDTH_64b, XED_ADDRESS_WIDTH_64b);
}

xed_error_enum_t decode(xed_decoded_inst_t *xedd, void *ip) //解析这个ip对应的指令的类型
{
        if((uint64_t)ip > program_fini){
                return XED_ERROR_NONE;
        }
        uint8_t* phy_ip = (uint64_t)ip + program_phy_base;
        /*printf("ip = %llx, phy_ip = %llx, program_base = %llx\n",(uint64_t)ip,phy_ip,program_phy_base);
        printf(" Inst: ");
    for(int i=0;i<8;i++){
        printf("%02x ",*((unsigned char*)(phy_ip+i)));
    }
    printf("\n");*/


        xed_decoded_inst_zero_set_mode(xedd, &xedState);
        if(XED_ERROR_NONE != xed_decode(xedd, (const xed_uint8_t*)(phy_ip), MAX_BYTES_TO_DECODE)) {
                return XED_ERROR_GENERAL_ERROR;
        }
        return XED_ERROR_NONE;
}

INST_TYPE is_store_op(void* ip){ //解析这个ip的指令，确定是不是store指令
//return true;
        xed_decoded_inst_t xedd;
        if(XED_ERROR_NONE != decode(&xedd, ip)) {
                printf("Warning: Failed to disassemble instruction\n");
                return unknow_inst;
        }
        xed_uint_t numMemOps = xed_decoded_inst_number_of_memory_operands(&xedd);
        if (numMemOps != 1) {
                printf("Warning: numMemOps != 1 (=%d)\n",numMemOps);
                return unknow_inst;
        }
        xed_bool_t isOpRead =  xed_decoded_inst_mem_read(&xedd, 0);
        xed_bool_t isOpWritten =  xed_decoded_inst_mem_written(&xedd, 0);
        uint32_t accessLen = xed_decoded_inst_get_memory_operand_length(&xedd, 0);
        if(isOpWritten){
                return store_inst;
        }else if(isOpRead){
                return load_inst;
        }else{
                return unknow_inst;
        }
}

uint8_t* find_pre_ip(uint8_t* wp_ip){ //找到前一条指令
        uint64_t ip=(uint64_t)wp_ip;
        std::unordered_map<uint64_t,uint64_t>::iterator it= pc_list.find(ip);
        if(it != pc_list.end()){
                ip = it->second;
        }else{
                printf("Warning: ip:%llx not in elf_list\n",ip);
                ip=0;
        }
        return (uint8_t*)ip; 
}


//void watchpoint_handler(int sig_num,siginfo_t *sig_info,void *context){
void watchpoint_handler(){
                //std::cout<<"Thread "<<gettid()<<" trap at watchpoint"<<std::endl;
                //ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_DISABLE,0);
                void* watchpoint_buffer;
                uint64_t next_offset_wp;
                int wp_fd;
                bool store_op;
                uint8_t* wp_ip=(uint8_t*)malloc(sizeof(uint64_t));
                uint64_t cur_ip;

                //if(sig_info->si_fd == sample_data.wp_r_fd){
                        watchpoint_buffer=sample_data.wp_r_buffer;
                        next_offset_wp=sample_data.wp_r_next_offset;
                        wp_fd=sample_data.wp_r_fd;
                /*}else{
                        int ret = ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);
                        assert(ret==0);
                        return;
                }*/

                //Read method1:
                bool read_buffer_success=true;
                struct perf_event_header hdr;
                if (readBuffer(watchpoint_buffer, &hdr, sizeof(struct perf_event_header)) < 0){
                        printf("watchpoint read header failed\n");
                        read_buffer_success=false;
                }else{
                        //printf("hdr->size: %u\n",hdr.size);
                }
                if ((read_buffer_success)&&(readBuffer(watchpoint_buffer, &wp_ip, sizeof(uint64_t)) < 0)){
                        printf("watchpoint read ip failed\n");
                        read_buffer_success=false;
                }else{
                        //printf("pc: wp_ip:%llx\n",wp_ip);
                }
                //debug:read addr of wp
                uint64_t wp_addr_check;
                if ((read_buffer_success)&&(readBuffer(watchpoint_buffer, &wp_addr_check, sizeof(uint64_t)) < 0)){
                        printf("watchpoint read addr failed\n");
                        read_buffer_success=false;
                }else{
                        //printf("wp addr: %llx\n",wp_addr_check);
                }
                if((!read_buffer_success)||(wp_addr_check!=sample_data.target_addr)){
                        printf("watchpoint read new wp failed\n");
                        first_store_arrive=false;
                        first_load_arrive=false;
                        //int ret = ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);
                        //assert(ret==0);
                        return;
                }

                //TODO:ip maybe wrong and cause segment fault (vaddr of which program?)

                //Read method2:
                /*struct sample{
                                struct perf_event_header header;
                                uint64_t ip;
                }* sample=(struct sample*)((uint8_t*)watchpoint_buffer+4096+next_offset_wp);
                struct perf_event_mmap_page* info=(struct perf_event_mmap_page*)watchpoint_buffer;
                sample_data.wp_r_next_offset=info->data_head%(RING_BUFFER_PAGES*4096);
                *wp_ip = sample->ip;
                cur_ip = *wp_ip;*/

                //printf("cur_ip, pre_ip:(%llx,%llx)\n",wp_ip,sample_data.pre_ip);

                //hdr.misc == sample->header.misc
                //if(hdr.misc & PERF_RECORD_MISC_EXACT_IP){
                if(hdr.misc & PERF_RECORD_MISC_EXACT_IP){
                        //printf("###precise ip: %llx\n",cur_ip);
                }else{
                        //printf("@@@no precise: %llx\n",cur_ip);
                }

                uint8_t* pre_wp_ip = find_pre_ip(wp_ip); //找到前一条指令
                if((uint64_t)pre_wp_ip == 0){
                        first_store_arrive=false;
                        first_load_arrive=false;
                        //int ret = ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);
                        //assert(ret==0);
                        return;
                }
                INST_TYPE inst_type;
                inst_type = is_store_op(pre_wp_ip); //前一条是不是store
                /*int ip_dis=0;
                while((unknow_inst==inst_type)&&(ip_dis<5)){
                        cur_ip--;
                        ip_dis++;
                        inst_type = is_store_op(&cur_ip);
                }*/


                if(inst_type==store_inst){store_op=true;}
                else if(inst_type==load_inst){store_op=false;}
                else{
                        printf("%llx not load or store.\n",pre_wp_ip);
                        //int ret = ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);
                        //assert(ret==0);
                        return;
                }
                //printf("Watchpoint: ip %llx %c addr %llx\n",pre_wp_ip,store_op?'w':'r',wp_addr_check);

                if(detector==DEADSTORE){ //要识别的是deadstore指令
                        if(!first_store_arrive){ //之前还没有监控到store指令
                                if(store_op){
                                                sample_data.pre_ip = (uint64_t)pre_wp_ip;
                                                first_store_arrive=true;
                                                //printf("first store has arrived, wait for next access\n");
                                }else{
                                        //int ret = ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);
                                        //assert(ret==0);
                                        return;
                                }
                        }else{//first store has arrived,this is the second access to this addr
                                //printf("next access arrived\n");
                                if(store_op){ //连续的store->deadstore
                                        first_store_arrive=false;
                                        dead_amount++;
                                        pthread_mutex_lock(&cnt_mutex);
                                        access_cnt=0;
                                        pthread_mutex_unlock(&cnt_mutex);
                                        time_t time_get = time(NULL);
                                        printf("RECORD_NEW_DEADSTORE %llx %llx\n",pre_wp_ip,sample_data.pre_ip);
                                }else{ //store后紧跟load，不是deadstore
                                        first_store_arrive=false;
                                }
                        }
                //TODO：silent store和load的处理
                }else if(detector==SILENTSTORE){
                        if(!first_store_arrive){
                                //record first pc
                                first_store_arrive=true;
                        }else{
                                int find_result=rand()%2;
                                //record second pc & cmp
                                if(find_result){
                                        first_store_arrive=false;
                                        pthread_mutex_lock(&cnt_mutex);
                                        access_cnt=0;
                                        pthread_mutex_unlock(&cnt_mutex);
                                }else{
                                        first_store_arrive=false;
                                }
                        }
                }else if(detector==SILENTLOAD){
                        if(!first_load_arrive){
                                //record first pc
                                first_load_arrive=true;
                        }else{
                                int find_result=rand()%2;
                                //record second pc & cmp
                                if(find_result){
                                        first_load_arrive=false;
                                        pthread_mutex_lock(&cnt_mutex);
                                        access_cnt=0;
                                        pthread_mutex_unlock(&cnt_mutex);
                                }else{
                                        first_load_arrive=false;
                                }
                        }
                }

                //ioctl(sample_data.wp_r_fd,PERF_EVENT_IOC_ENABLE,0);
                //printf("watchpoint_handler finish\n");
}


//采样完成后的信号处理函数
//void sample_handler(int sig_num,siginfo_t *sig_info,void *context)
void sample_handler()
{
                //std::cout<<"Thread "<<gettid()<<" get new sample "<<wp_state<<std::endl;
                int ret = ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
                assert(ret==0);

                /*if(sig_info->si_code<0){ //signal si_code < 0 indicates not from kernel(check if signal generated by kernel for profiling)
                        int ret = ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
                        assert(ret==0);
                        return;
                }*/

        struct perf_event_mmap_page* rinfo=(perf_event_mmap_page*)rbuf;
    next_offset=rinfo->data_head%(RING_BUFFER_PAGES*4096);
        if(next_offset == 0){next_offset=RING_BUFFER_PAGES*4096;}
    uint64_t offset=4096+next_offset-sizeof(perf_my_sample);

    struct perf_my_sample* sample=(perf_my_sample*)((uint8_t*)rbuf+offset); //获得记录的sample信息
    if(sample->header.type==PERF_RECORD_SAMPLE)
    {
                                last_sample_t=clock();
                        //printf("PMU sample_ip: %lx, addr: %lx\n",sample->ip,sample->addr);
                                sample_data.addr = sample->addr;
                                sample_data.ip = sample->ip;

                                if(sample->addr == 0){
                                        int ret = ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
                                        assert(ret==0);
                                        return;
                                }

                                if(phase_get_memory){ //要做的事是收集访存
                                        sample_memory_access_cnt++;
                                        //uint64_t func_ip = get_func_ip(sample->ip);
                                        //memoryFile<<func_ip<<" "<<sample->addr<<std::endl;
                                        //memoryFile<<sample->addr<<std::endl;
                                }else if(is_target_func(sample->ip)){ //要做的事是监控deadstore
                                        pthread_mutex_lock(&cnt_mutex);

                                        double probabilityToReplace =  1.0/(1.0 + (double)access_cnt); //是否要替换debug_register监控对象
                                        access_cnt++;
                                        double randValue;

                                        randValue = (rand()%10000)/(double)10000;
                                        //printf("replace: randv=%lf p=%lf\n",randValue,probabilityToReplace);
                                        if(randValue <= probabilityToReplace){
                                                pthread_mutex_unlock(&cnt_mutex);
                                                printf("REPLACY_DEBUG_REGISTER\n");
                                                if(wp_state==WP_RUN){
                                                        //printf("Thread PMU: wait to lock wp_mutex.\n");
                                                        pthread_mutex_lock(&wp_mutex);
                                                        //printf("Thread PMU: lock wp_mutex succeed.\n");
                                                        close_bp(); //先关闭之前的debug_register
                                                        CREATE_RET ret=create_bp(sample->addr); //重新开一个
                                                        if(ret!=CREATE_SUCCESS){
                                                                printf("[replace:WP_RUN] create_bp error: %d.\n",ret);
                                                        }
                                                        pthread_mutex_unlock(&wp_mutex);
                                                        //printf("Thread PMU: unlock wp_mutex.\n");
                                                }else{
                                                        printf("Thread PMU: create when debug register off.\n");
                                                        //printf("Thread PMU: wait to lock wp_mutex.\n");
                                                        pthread_mutex_lock(&wp_mutex);
                                                        //printf("Thread PMU: lock wp_mutex succeed.\n");
                                                        CREATE_RET ret=create_bp(sample->addr);
                                                        if(ret!=CREATE_SUCCESS){
                                                                printf("[replace:WP_CLEARED] create_bp error: %d.\n",ret);
                                                        }
                                                        pthread_mutex_unlock(&wp_mutex);
                                                        //printf("Thread PMU: unlock wp_mutex.\n");
                                                }
                                        }else{
                                                pthread_mutex_unlock(&cnt_mutex);
                                                if(wp_state==WP_CLEARED){
                                                        //printf("Thread PMU: wait to lock wp_mutex.\n");
                                                        pthread_mutex_lock(&wp_mutex);
                                                        //printf("Thread PMU: lock wp_mutex succeed.\n");
                                                        CREATE_RET ret=create_bp(sample->addr);
                                                        if(ret!=CREATE_SUCCESS){
                                                                printf("[not replace] create_bp error: %d.\n",ret);
                                                        }
                                                        pthread_mutex_unlock(&wp_mutex);
                                                        //printf("Thread PMU: unlock wp_mutex.\n");
                                                }
                                        }
                                }
    }else{
                printf("sample->header.type != PERF_RECORD_SAMPLE(%d) [==%d]\n",PERF_RECORD_SAMPLE,sample->header.type);
                printf("sample{'type=%lu','misc=%lu','size=%u'}\n",sample->header.type,sample->header.misc,sample->header.size);
        }
    //struct perf_event_mmap_page* rinfo=(perf_event_mmap_page*)rbuf;
    //next_offset=rinfo->data_head%(RING_BUFFER_PAGES*4096);

        ret = ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        assert(ret==0);
        //printf("sample handler finish\n");
}


bool cmp(elf_parser::symbol_t a,elf_parser::symbol_t b){
        if(a.symbol_value < b.symbol_value){
                return true;
        }else{
                return false;
        }
}

void symbol_sort(std::vector<elf_parser::symbol_t> &symbols){
        sort(symbols.begin(),symbols.end(),cmp);

}

void* wait_thread_sample(void* arg){ //等待debug_register的信号，执行sample_handler函数
        std::cout<<"Thread "<<gettid()<<" wait for sample signal."<<std::endl;
        int sig;
        int *sigptr=&sig;
        siginfo_t *info;
        struct timespec timeout;
        timeout.tv_sec = 5; 
        timeout.tv_nsec = 0;
        while(!target_exit){ //main函数中未设置终止
                sigtimedwait(&receive_set_mem,info,&timeout);
                //sigwait(&receive_set_mem,sigptr);
                //std::cout<<"Thread PMU: in while, wait for new sample.\n";
                //if(*sigptr == MEM_SIGNAL){
                        //std::cout<<"Thread Sample: new signal:"<<*sigptr<<std::endl;
                        sample_handler();
                //}else{
                //      std::cout<<"Thread Sample: wrong signal:"<<*sigptr<<std::endl;
                //}
        }
}


void* wait_thread_wp(void* arg){//等待PMU的信号，执行watchpoint_handler函数
        std::cout<<"Thread "<<gettid()<<" wait for watchpoint signal."<<std::endl;
        int sig;
        int *sigptr=&sig;
        siginfo_t *info;
        struct timespec timeout;
        timeout.tv_sec = 5; 
        timeout.tv_nsec = 0;
        while(!target_exit){ //main函数中未设置终止
                sigtimedwait(&receive_set_wp,info,&timeout);
                //sigwait(&receive_set_wp,sigptr);
                //std::cout<<"Thread Wp: new signal:"<<*sigptr<<std::endl;
                if(pthread_mutex_trylock(&wp_mutex)==0){
                        //printf("Thread Wp: lock wp_mutex succeed.\n");
                        if(wp_state == WP_RUN){
                                watchpoint_handler(); //观察点
                        }
                        pthread_mutex_unlock(&wp_mutex);
                        //printf("Thread Wp: unlock wp_mutex.\n");
                }
        }
}

//之前debug使用的
void workload()
{
  for(int j=0;j<100;j++){
    for(int i=0;i<10000;i++)
    {   
      data_list[i]++;
    }   
  }
}




int main(int argc,char** argv)
{
                XedInit();
                srand((unsigned)time(NULL));
                pthread_mutex_init( &wp_mutex, NULL ); 
                pthread_mutex_init( &cnt_mutex, NULL ); 
                pid_target=atoi(argv[1]);
                std::cout<<"pid:"<<pid_target<<std::endl;
                std::string program((std::string)argv[2]); //binary
                std::cout<<"binary file:"<<program<<std::endl;
                std::cout<<"label file:"<<argv[3]<<std::endl;
                std::cout<<"output file:"<<argv[4]<<std::endl;
                std::cout<<"assFile:"<<argv[5]<<std::endl;
                elf_parser::Elf_parser elf_parser(program); //解析binary
                syms = elf_parser.get_symbols();
                symbol_sort(syms);
                program_phy_base = elf_parser.get_phy_base();
                program_fini = elf_parser.get_fini();
                program_init = elf_parser.get_init();
                std::ifstream labelFile(argv[3]);
                memoryFile.open(argv[4]);
                std::ifstream assFile(argv[5]);
                int detect_time = atoi(argv[7]);
                phase_get_memory = atoi(argv[8]);
                detector = DEADSTORE; //DEADSTORE;
                int pre_label;
                uint64_t func_addr;
                std::string bench_name;
                while(labelFile>>pre_label){
                        labelFile>>func_addr;
                        labelFile>>bench_name;
                        funcLabelList.insert(std::pair<uint64_t,int>(func_addr,pre_label)); //读取预测结果，是否过滤
                        //std::cout<<"funcLabelList add "<<func_addr<<" "<<pre_label<<std::endl;
                }
                printf("labelFuncList size: %d\n",funcLabelList.size());

                std::string assline;
                uint64_t prePc=0;
                while(getline(assFile,assline)){ //读取汇编文件，获得整个程序的指令序列
                        //std::cout<<assline<<std::endl;
                        int index = assline.find(":");  
                        if(index == -1){continue;}  
                        if(assline[0] != ' '){continue;}
                        std::string assPc;
                        if(assline[1] != ' '){
                                assPc = assline.substr(1,index);
                        }else{
                                assPc = assline.substr(2,index);
                        }
        
                        char *ptr;
                        uint64_t assPc_int = strtoull(assPc.c_str(),&ptr,16);
                        //std::cout<<std::hex<<assPc_int<<std::endl;
                        pc_list.insert(std::pair<uint64_t,uint64_t>(assPc_int,prePc)); //记录每条指令和它的前一条指令，方便后续查找每天指令的前序指令
                        prePc = assPc_int;
                }

                printf("pcListSize: %d  elfFuncListSize: %d\n",pc_list.size(),syms.size()); 

                wp_state=WP_CLEARED;

    struct perf_event_attr attr; //配置监控参数
    memset(&attr,0,sizeof(struct perf_event_attr));
    attr.size=sizeof(struct perf_event_attr);
    attr.type=PERF_TYPE_RAW;
        attr.config=(uint64_t)(0x82D0); //监控的类别0x82D0
        //attr.freq=1;
        //attr.sample_freq=atoi(argv[6]); //20000;

        attr.sample_period=atoi(argv[6]); //100000000;
    attr.sample_type=PERF_SAMPLE_IP|PERF_SAMPLE_ADDR; //每个sample需要记录的信息
                attr.exclude_kernel=1;
                attr.precise_ip = 3; //监控精确度
    attr.disabled=1; //先不开启监控
        fd=perf_event_open(&attr,pid_target,-1,-1,0); //绑到fd上
    if(fd<0)
    {
        perror("Cannot open perf fd!");
        return 1;
    }
    //创建1+16页共享内存，应用程序只读，读取fd产生的内容
    rbuf=mmap(0,(1+RING_BUFFER_PAGES)*4096,PROT_READ,MAP_SHARED,fd,0); //分配记录的buffer
    if(rbuf<0)
    {
        perror("Cannot mmap!");
        return 1;
    }
    fcntl(fd,F_SETFL,O_RDWR|O_NONBLOCK|O_ASYNC);
    fcntl(fd,F_SETSIG,MEM_SIGNAL);
    fcntl(fd,F_SETOWN,getpid());

/*    //set memory_access signal
                sigset_t block_mask;
                sigfillset(&block_mask); //put all signal in block_mask

    struct sigaction sig;
    memset(&sig,0,sizeof(struct sigaction));
    sig.sa_sigaction=sample_handler;
                sig.sa_mask=block_mask;
    sig.sa_flags=SA_SIGINFO | SA_ONSTACK;
    if(sigaction(MEM_SIGNAL,&sig,0)<0)
    {
        perror("Cannot sigaction");
        return 1;
    }

        //set watchpoint signal
        struct sigaction sig_wp;
        memset(&sig_wp,0,sizeof(struct sigaction));
        sig_wp.sa_sigaction=watchpoint_handler;
        sig_wp.sa_mask=block_mask;
        //sig_r_wp.sa_flags=SA_SIGINFO | SA_RESTART | SA_NODEFER | SA_ONSTACK; //SA_NODEFER:sig handle中不阻塞WP_SIGNAL信号
        sig_wp.sa_flags=SA_SIGINFO | SA_RESTART | SA_ONSTACK; 
        if(sigaction(WP_SIGNAL,&sig_wp,0)<0){
                perror("Create WP_SIGNAL failed");
                return 1;
        }
*/

        sigemptyset(&receive_set_mem);
        sigemptyset(&receive_set_wp);
        sigemptyset(&receive_set);
        sigaddset(&receive_set_mem,MEM_SIGNAL);
        sigaddset(&receive_set_wp,WP_SIGNAL);
        sigaddset(&receive_set,MEM_SIGNAL);
        sigaddset(&receive_set,WP_SIGNAL);
        pthread_sigmask(SIG_SETMASK,&receive_set,NULL);

        int ret;
        ret = pthread_create( &th1, NULL, wait_thread_sample, NULL );
        if( ret != 0 ){  
                printf( "Create thread error!\n");  
                return -1;  
        }
        ret = pthread_create( &th2, NULL, wait_thread_wp, NULL );
        if( ret != 0 ){  
                printf( "Create thread error!\n");  
                return -1;  
        }

        //开始监测
        ioctl(fd,PERF_EVENT_IOC_RESET,0);
        ioctl(fd,PERF_EVENT_IOC_ENABLE,0);

        printf("start monitor\n");
        clock_t time_start=clock();
        //workload();
        while(!target_exit){
                clock_t time_now=clock();
                //std::cout<<"main wait:"<<target_exit<<" "<<last_sample_t<<std::endl;;
                //if(time_now - last_sample_t > 3000000){
                if(time_now - time_start > detect_time*1000000){ //长时间没检测到新的sample
                        target_exit=true;
                        printf("no new sample\n");
                        break;
                }
        };
        printf("finish monitor\n");
        pthread_join( th1, NULL );  //TODO: thread block at sigwait, will not exit
        pthread_join( th2, NULL );  //TODO: thread block at sigwait, will not exit
        printf("dead amount: %lu, sample_memory_access_cnt=%lu\n",dead_amount,sample_memory_access_cnt);

        //停止监测
    ioctl(fd,PERF_EVENT_IOC_DISABLE,0);
    munmap(rbuf,(1+RING_BUFFER_PAGES)*4096);
    close(fd);
        pthread_mutex_destroy( &wp_mutex );  
        pthread_mutex_destroy( &cnt_mutex ); 
        printf("ALL FINISH\n"); 
    return 0;
}
