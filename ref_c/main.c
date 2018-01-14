#include <stdio.h>

FILE *fp_fprintf_debug;
static unsigned int s_fprintf_debug_init_u32 = 0;

static void convolution_ref_c(float *p_out_f32, float *p_in_f32, float *p_weights_f32);

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

    printf("yolo reference C code by Hyuk Lee\n");

    fp_fprintf_debug = fopen("ref_c_debug.txt", "w");

    /* read input data (letterbox_image currently) */
    fp = fopen("yolo_image_sized.bin", "rb");
    if(NULL == fp)
    {
        printf("read input data fopen error\n");
        return -1;
    }
    fread(sa_image_sized_f32, 608 * 608 * 3, sizeof(float), fp);
    fclose(fp);
    /* load weights */
    fp = fopen("yolo_cpu_weights_b_0_g_0_3x3x3x32.bin", "rb");
    if(NULL == fp)
    {
        printf("read weights fopen error\n");
        return -1;
    }
    fread(sa_weights_l00_f32, 3 * 3 * 3 * 32, sizeof(float), fp);
    fclose(fp);
    /* load mean */
    fp = fopen("yolo_cpu_rolling_mean_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read mean fopen error\n");
        return -1;
    }
    fread(sa_mean_l00_f32, 32, sizeof(float), fp);
    fclose(fp);
    /* load variance */
    fp = fopen("yolo_cpu_rolling_variance_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read variance fopen error\n");
        return -1;
    }
    fread(sa_variance_l00_f32, 32, sizeof(float), fp);
    fclose(fp);
    /* load scale */
    fp = fopen("yolo_cpu_scales_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read scale fopen error\n");
        return -1;
    }
    fread(sa_scale_l00_f32, 32, sizeof(float), fp);
    fclose(fp);
    /* load bias */
    fp = fopen("yolo_cpu_biases_b_1_32x608x608.bin", "rb");
    if(NULL == fp)
    {
        printf("read bias fopen error\n");
        return -1;
    }
    fread(sa_bias_l00_f32, 32, sizeof(float), fp);
    fclose(fp);

    convolution_ref_c(sa_out_l00_f32, sa_image_sized_f32, sa_weights_l00_f32);

    fclose(fp_fprintf_debug);

    /* read ref data layer 0 */
    fp = fopen("yolo_cpu_layer_0.bin", "rb");
    if(NULL == fp)
    {
        printf("read ref data l00 fopen error\n");
        return -1;
    }
    fread(sa_ref_l00_f32, 608 * 608 * 32, sizeof(float), fp);
    fclose(fp);

    for(k = 0; k < 32; k++)
    {
        for(j = 0; j < 608; j++)
        {
            for(i = 0; i < 608; i++)
            {
                if(sa_out_l00_f32[i + j * 608 + k * 608 * 608] != sa_ref_l00_f32[i + j * 608 + k * 608 * 608])
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
    unsigned int i, j, ci, co, kw, kh;
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
                            src_f32 = p_in_f32[ci * 608 * 608 + j * 608 + i + kh * 3 + kw];
                            wei_f32 = p_weights_f32[co * 3 * 3 * 3 + ci * 3 * 3 + kh * 3 + kw];
                            acc_f32 += src_f32 * wei_f32;
                            if(co == 0)
                            {
                                fprintf(fp_fprintf_debug, "kw: %d, kh: %d, ci: %d, i: %d, j: %d, in: %f, wei: %f, acc: %f\n", kw, kh, ci, i, j, src_f32, wei_f32, acc_f32);
                            }
                        }
                    }
                }
                p_out_f32[co * 608 * 608 + j * 608 + i] = acc_f32;
            }
        }
    } 
}

