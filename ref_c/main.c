#include <stdio.h>
#include <math.h>
#include <time.h>

#define DEBUG_WRITING (0)

#if (1 == DEBUG_WRITING)
FILE *fp_fprintf_debug;
static unsigned int s_fprintf_debug_init_u32 = 0;
#endif

static void convolution_ref_c(float *p_out_f32, float *p_in_f32, float *p_weights_f32);
static void normalize_cpu(float *x, float *mean, float *variance, int filters, int spatial);
static void scale_bias(float *output, float *scales, int n, int size);
static void add_bias(float *output, float *biases, int n, int size);

int main(void)
{
    FILE *fp;
    static float sa_image_sized_f32[608 * 608 * 32];
    static float sa_out_l00_f32[608 * 608 * 32];
    static float sa_weights_l00_f32[3 * 3 * 3 * 32];
    static float sa_mean_l00_f32[32];
    static float sa_variance_l00_f32[32];
    static float sa_scale_l00_f32[32];
    static float sa_bias_l00_f32[32];
    static float sa_ref_l00_f32[608 * 608 * 32];
    int i, j, k;
    size_t fread_return;
    clock_t clk_srt, clk_end;

    printf("yolo reference C code by Hyuk Lee\n");

#if (1 == DEBUG_WRITING)
    fp_fprintf_debug = fopen("ref_c_debug.txt", "w");
#endif

    /* read input data (letterbox_image currently) */
    fp = fopen("yolo_image_sized.bin", "rb");
    if(NULL == fp)
    {
        printf("read input data fopen error\n");
        return -1;
    }
    fread_return = fread(sa_image_sized_f32, 608 * 608 * 3, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load weights */
    fp = fopen("yolo_cpu_weights_b_0_g_0_3x3x3x32.bin", "rb");
    if(NULL == fp)
    {
        printf("read weights fopen error\n");
        return -1;
    }
    fread_return = fread(sa_weights_l00_f32, 3 * 3 * 3 * 32, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load mean */
    fp = fopen("yolo_cpu_rolling_mean_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read mean fopen error\n");
        return -1;
    }
    fread_return = fread(sa_mean_l00_f32, 32, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load variance */
    fp = fopen("yolo_cpu_rolling_variance_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read variance fopen error\n");
        return -1;
    }
    fread_return = fread(sa_variance_l00_f32, 32, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);
    /* load scale */
    fp = fopen("yolo_cpu_scales_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read scale fopen error\n");
        return -1;
    }
    fread_return = fread(sa_scale_l00_f32, 32, sizeof(float), fp);
    fclose(fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    /* load bias */
    fp = fopen("yolo_cpu_biases_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read bias fopen error\n");
        return -1;
    }
    fread_return = fread(sa_bias_l00_f32, 32, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);

    clk_srt = clock();
    convolution_ref_c(sa_out_l00_f32, sa_image_sized_f32, sa_weights_l00_f32);
    clk_end = clock();
    printf("l00 convolution: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);
    clk_srt = clock();
    normalize_cpu(sa_out_l00_f32, sa_mean_l00_f32, sa_variance_l00_f32, 32, 608 * 608);
    clk_end = clock();
    printf("l00 normalize: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);
    clk_srt = clock();
    scale_bias(sa_out_l00_f32, sa_scale_l00_f32, 32, 608 * 608);
    clk_end = clock();
    printf("l00 scale_bias: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);
    clk_srt = clock();
    add_bias(sa_out_l00_f32, sa_bias_l00_f32, 32, 608 * 608);
    clk_end = clock();
    printf("l00 add_bias: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);

#if (1 == DEBUG_WRITING)
    fclose(fp_fprintf_debug);
#endif

    /* read ref data layer 0 */
    fp = fopen("yolo_cpu_layer_0.bin", "rb");
    if(NULL == fp)
    {
        printf("read ref data l00 fopen error\n");
        return -1;
    }
    fread_return = fread(sa_ref_l00_f32, 608 * 608 * 32, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);

    for(k = 0; k < 32; k++)
    {
        for(j = 0 + 1; j < 608 - 1; j++)
        {
            for(i = 0 + 1; i < 608 - 1; i++)
            {
                if(fabsf(sa_out_l00_f32[i + j * 608 + k * 608 * 608] - sa_ref_l00_f32[i + j * 608 + k * 608 * 608]) > 0.000001f)
                {
                    printf("layer_0_f32 mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_out_l00_f32[i + j * 608 + k * 608 * 608], sa_ref_l00_f32[i + j * 608 + k * 608 * 608]);
                }
            }
        }
    }

    return 0;
}

static void convolution_ref_c(float *p_out_f32, float *p_in_f32, float *p_weights_f32)
{
    int i, j, ci, co, kw, kh;
    float src_f32;
    float wei_f32;
    float acc_f32;

    for(co = 0; co < 32; co++)
    {
        for(j = 0; j < 608; j++)
        {
            for(i = 0; i < 608; i++)
            {
                acc_f32 = 0.0f;
                for(ci = 0; ci < 3; ci++)
                {
                    for(kh = 0; kh < 3; kh++)
                    {
                        for(kw = 0; kw < 3; kw++)
                        {
                            src_f32 = p_in_f32[ci * 608 * 608 + (j - 1 + kh) * 608 + (i - 1) + kw];
                            wei_f32 = p_weights_f32[co * 3 * 3 * 3 + ci * 3 * 3 + kh * 3 + kw];
                            acc_f32 += src_f32 * wei_f32;
#if (1 == DEBUG_WRITING)
                            if((co == 0) && (ci == 0) && (j < 20) || ((j > 60) && (j < 80)))
                            {
                                fprintf(fp_fprintf_debug, "kw: %d, kh: %d, ci: %d, i: %d, j: %d, in: %f, wei: %f, acc: %f\n", kw, kh, ci, i, j, src_f32, wei_f32, acc_f32);
                            }
#endif
                        }
                    }
                }
                p_out_f32[co * 608 * 608 + j * 608 + i] = acc_f32;
            }
        }
    } 
}

static void normalize_cpu(float *x, float *mean, float *variance, int filters, int spatial)
{
    int f, i;

    for(f = 0; f < filters; ++f)
    {
        for(i = 0; i < spatial; ++i)
        {
            int index = f * spatial + i;
            x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + .000001f);
        }
    }
}

static void scale_bias(float *output, float *scales, int n, int size)
{
    int i, j;

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < size; j++)
        {
            output[i * size + j] *= scales[i];
        }
    }
}

static void add_bias(float *output, float *biases, int n, int size)
{
    int i, j;

    for(i = 0; i < n; i++)
    {
        for(j = 0; j < size; j++)
        {
            output[i * size + j] += biases[i];
            /* LEAKY ACTIATION */
            output[i * size + j] = (output[i * size + j] > 0) ? output[i * size + j] : 0.1f * output[i * size + j];
        }
    }
}

