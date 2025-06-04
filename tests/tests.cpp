#include "../scan.h"
#include "gtest/gtest.h"


void init_array1(float *arr, int size)
{
    for (unsigned int i = 0; i < size; i++)
    {
        arr[i] = (float)(i % 100) + 0.1f;
    }
}


void init_array2(float *arr, int size)
{
    for (unsigned int i = 0; i < size; i++)
    {
        arr[i] = (float)(i % 100) + 1.11f;
    }
}


class CudaScanTest : public ::testing::TestWithParam<int>
{
   protected:
    CudaScanTest() {}

    ~CudaScanTest() override
    {
        free(h_in_);
        free(h_out_);
        free(h_out_cpu_);
    }

    void SetUp() override
    {
        size_ = GetParam();
        std::cout << "    ArrayTestFixture::SetUp() for size: " << size_ << std::endl;

        unsigned int array_size_bytes_ = size_ * sizeof(float);
        h_in_ = (float *)malloc(array_size_bytes_);
        h_out_ = (float *)malloc(array_size_bytes_);
        h_out_cpu_ = (double *)malloc(size_ * sizeof(double));

        init_array2(h_in_, size_);

        h_out_cpu_[0] = h_in_[0];

        for (unsigned int i = 1; i < size_; i++)
        {
            h_out_cpu_[i] = h_out_cpu_[i - 1] + h_in_[i];
        }
    }

    void TearDown() override {}

    float *h_in_ = nullptr;
    float *h_out_ = nullptr;
    double *h_out_cpu_ = nullptr;
    int size_ = 0;
    int array_size_bytes_ = 0;
};

TEST(MyLibSuite, AddFunction)
{
    EXPECT_EQ(2 + 3, 5);
    EXPECT_FLOAT_EQ(2.3, 2.3);
}


TEST_P(CudaScanTest, TestKernel1)
{
    call_scan_kernel1(h_in_, h_out_, size_);
    for (unsigned int i = 0; i < size_; i++)
    {
        EXPECT_FLOAT_EQ(h_out_[i], (float)h_out_cpu_[i]);
    }
}


TEST_P(CudaScanTest, TestKernel2)
{
    call_scan_kernel2(h_in_, h_out_, size_);
    for (unsigned int i = 0; i < size_ - 1; i++)
    {
        EXPECT_FLOAT_EQ(h_out_[i + 1], (float)h_out_cpu_[i]);
    }
}


TEST_P(CudaScanTest, TestKernel3)
{
    call_scan_kernel3(h_in_, h_out_, size_);
    for (unsigned int i = 0; i < size_ - 1; i++)
    {
        EXPECT_FLOAT_EQ(h_out_[i + 1], (float)h_out_cpu_[i]);
    }
}


INSTANTIATE_TEST_SUITE_P(ScanRandomTests, CudaScanTest,
                         ::testing::Values(1, 2, 5, 10, 16, 19, 27, 46, 74, 123, 189, 346, 786,
                                           1024, 2048, 4657, 14765, 146873, (1 << 16),
                                           (1 << 17) - 38, (1 << 18) + 49, (1 << 20) + 1,
                                           (1 << 21) - 1, (1 << 22) + 234, (1 << 25) + 78,
                                           (1 << 29) - 134987));
