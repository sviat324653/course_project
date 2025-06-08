#include <stdio.h>

int reduction_test();
int scan_test();
int stream_compaction_test();
int sort_test();

int main()
{
    printf("\n\t\t\t----------REDUCTION----------\n\n\n");
    reduction_test();
    printf("\n\t\t\t------------SCAN-------------\n\n\n");
    scan_test();
    printf("\n\t\t\t------STREAM COMPACTION------\n\n\n");
    stream_compaction_test();
    printf("\n\t\t\t------------SORT-------------\n\n\n");
    sort_test();
}
