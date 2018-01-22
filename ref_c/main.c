#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DEBUG_WRITING (1)

#define WID_SRC (768)
#define HEI_SRC (576)
#define CHN_SRC (3)
#define WID_L00 (608)
#define HEI_L00 (608)
#define CHN_L00 (32)
#define K_W_L00 (3)
#define K_H_L00 (3)
#define PAD_L00 (1)

#if (1 == DEBUG_WRITING)
FILE *fp_fprintf_debug;
#endif

static void convolution_ref_c(float *p_out_f32, float *p_in_f32, float *p_weights_f32);
static void normalize_cpu(float *x, float *mean, float *variance, int filters, int spatial);
static void scale_bias(float *output, float *scales, int n, int size);
static void add_bias(float *output, float *biases, int n, int size);

int main(void)
{
    FILE *fp;
    static float sa_image_sized_f32[WID_L00 * HEI_L00 * CHN_L00];
    static float sa_out_l00_f32[WID_L00 * HEI_L00 * CHN_L00];
    static float sa_weights_l00_f32[K_W_L00 * K_H_L00 * CHN_SRC * CHN_L00];
    static float sa_mean_l00_f32[CHN_L00];
    static float sa_variance_l00_f32[CHN_L00];
    static float sa_scale_l00_f32[CHN_L00];
    static float sa_bias_l00_f32[CHN_L00];
    static float sa_ref_l00_f32[WID_L00 * HEI_L00 * CHN_L00];
    int i, j, k;
    size_t fread_return;
    clock_t clk_srt, clk_end;

    printf("yolo reference C code by Hyuk Lee\n");

    memset(sa_out_l00_f32, 0, WID_L00 * HEI_L00 * CHN_L00 * sizeof(float));

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
    fread_return = fread(sa_image_sized_f32, WID_L00 * HEI_L00 * 3, sizeof(float), fp);
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
    fread_return = fread(sa_weights_l00_f32, K_W_L00 * K_H_L00 * CHN_SRC * CHN_L00, sizeof(float), fp);
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
    fread_return = fread(sa_mean_l00_f32, CHN_L00, sizeof(float), fp);
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
    fread_return = fread(sa_variance_l00_f32, CHN_L00, sizeof(float), fp);
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
    fread_return = fread(sa_scale_l00_f32, CHN_L00, sizeof(float), fp);
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
    fread_return = fread(sa_bias_l00_f32, CHN_L00, sizeof(float), fp);
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
    {
        FILE *fp;
        char buffer[100];

        sprintf(buffer, "yolo_convolution_out_ref_c_%dx%dx%d.bin", WID_L00, HEI_L00, CHN_L00);
        fp = fopen(buffer, "wb");
        if(NULL == fp)
        {
            printf("yolo_convolution_out_ref_c open error\n");
        }
        fwrite(sa_out_l00_f32, WID_L00 * HEI_L00 * CHN_L00, sizeof(float), fp);
        fclose(fp);
    }
    clk_srt = clock();
    normalize_cpu(sa_out_l00_f32, sa_mean_l00_f32, sa_variance_l00_f32, CHN_L00, WID_L00 * HEI_L00);
    clk_end = clock();
    printf("l00 normalize: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);
    clk_srt = clock();
    scale_bias(sa_out_l00_f32, sa_scale_l00_f32, CHN_L00, WID_L00 * HEI_L00);
    clk_end = clock();
    printf("l00 scale_bias: %f secs\n", (double)(clk_end - clk_srt) / CLOCKS_PER_SEC);
    clk_srt = clock();
    add_bias(sa_out_l00_f32, sa_bias_l00_f32, CHN_L00, WID_L00 * HEI_L00);
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
    fread_return = fread(sa_ref_l00_f32, WID_L00 * HEI_L00 * CHN_L00, sizeof(float), fp);
    if(sizeof(float) != fread_return)
    {
        printf("fread error\n");
        return -1;
    }
    fclose(fp);

    for(k = 0; k < CHN_L00; k++)
    {
        for(j = 0 + PAD_L00; j < HEI_L00 - PAD_L00; j++)
        {
            for(i = 0 + PAD_L00; i < WID_L00 - PAD_L00; i++)
            {
                if(fabsf(sa_out_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00] - sa_ref_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00]) > 0.000001f)
                {
                    printf("layer_0_f32 mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_out_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00], sa_ref_l00_f32[i + j * WID_L00 + k * WID_L00 * HEI_L00]);
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

    for(co = 0; co < CHN_L00; co++)
    {
        for(j = 0; j < HEI_L00; j++)
        {
            for(i = 0; i < WID_L00; i++)
            {
                acc_f32 = 0.0f;
                for(ci = 0; ci < 3; ci++)
                {
                    for(kh = 0; kh < 3; kh++)
                    {
                        for(kw = 0; kw < 3; kw++)
                        {
                            src_f32 = p_in_f32[ci * WID_L00 * HEI_L00 + (j - PAD_L00 + kh) * WID_L00 + (i - PAD_L00) + kw];
                            wei_f32 = p_weights_f32[co * K_W_L00 * K_H_L00 * CHN_SRC + ci * K_W_L00 * K_H_L00 + kh * K_W_L00 + kw];
                            acc_f32 += src_f32 * wei_f32;
#if (1 == DEBUG_WRITING)
                            if((co == 0) && (ci == 0) && ((j < 20) || ((j > 60) && (j < 80))))
                            {
                                fprintf(fp_fprintf_debug, "kw: %d, kh: %d, ci: %d, i: %d, j: %d, in: %f, wei: %f, acc: %f\n", kw, kh, ci, i, j, src_f32, wei_f32, acc_f32);
                            }
#endif
                        }
                    }
                }
                p_out_f32[co * WID_L00 * HEI_L00 + j * WID_L00 + i] = acc_f32;
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

