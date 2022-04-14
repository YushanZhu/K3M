
#include <pcap.h>

#include <stdio.h>

 

int main(int argc, char *argv[])

{

    pcap_t *handle;                 /* 会话句柄 */

    char *dev;                      /* 要监听设备 */

    char errbuf[PCAP_ERRBUF_SIZE];  /* 错误字符串 */

    struct bpf_program fp;          /* 过滤器 */

    char filter_exp[] = "port 80";  /* 过滤器表达式 */

    bpf_u_int32 mask;               /* 子网掩码 */

    bpf_u_int32 net;                /* IP地址 */

    struct pcap_pkthdr header;      /* pcap报头 */

    const u_char *packet;           /* 报文 */

 

    /* 设备定义 */

    dev = pcap_lookupdev(errbuf);

    if (dev == NULL) 

    {

        fprintf(stderr, "Couldn't find default device: %s\n", errbuf);

        return(2);

    }

    /* 设备属性查看 */

    if (pcap_lookupnet(dev, &net, &mask, errbuf) == -1) 

    {

        fprintf(stderr, "Couldn't get netmask for device %s: %s\n", dev, errbuf);

        net = 0;

        mask = 0;

    }

    /* 混杂模式打开会话 */

    handle = pcap_open_live(dev, BUFSIZ, 1, 1000, errbuf);

    if (handle == NULL) 

    {

        fprintf(stderr, "Couldn't open device %s: %s\n", dev, errbuf);

        return(2);

    }

    /* 编辑使用过滤器 */

    if (pcap_compile(handle, &fp, filter_exp, 0, net) == -1) 

    {

        fprintf(stderr, "Couldn't parse filter %s: %s\n", filter_exp, pcap_geterr(handle));

        return(2);

    }

    if (pcap_setfilter(handle, &fp) == -1) 

    {

        fprintf(stderr, "Couldn't install filter %s: %s\n", filter_exp, pcap_geterr(handle));

        return(2);

    }

    /* 抓包单个数据包 */

    packet = pcap_next(handle, &header);

    /* 输出其长度 */

    printf("Jacked a packet with length of [%d]\n", header.len);

    /* 关闭会话 */

    pcap_close(handle);

    return(0);

}
