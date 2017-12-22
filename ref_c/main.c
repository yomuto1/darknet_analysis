#include <stdio.h>

int main(void)
{
    FILE *fp;
    static float sa_image_sized_f32[608 * 608 * 32];
    static float sa_out_l00_f32[608 * 608 * 32];
    static float sa_layer_0_f32[608 * 608 * 32];
    int i, j, k;

    printf("yolo reference C code by Hyuk Lee\n");

    /* read input data (letterbox_image currently) */
    fp = fopen("yolo_image_sized.bin", "rb");
    if(NULL == fp)
    {
        printf("read input data fopen error\n");
        return -1;
    }
    fread(sa_image_sized_f32, 608 * 608 * 3, sizeof(float), fp);
    fclose(fp);

    convolution_ref_c(sa_out_l00_f32, sa_image_sized_f32);

    /* read ref data layer 0 */
    fp = fopen("yolo_cpu_layer_0.bin", "rb");
    if(NULL == fp)
    {
        printf("read ref data l00 fopen error\n");
        return -1;
    }
    fread(sa_layer_0_f32, 608 * 608 * 32, sizeof(float), fp);
    fclose(fp);

    for(k = 0; k < 32; k++)
    {
        for(j = 0; j < 608; j++)
        {
            for(i = 0; i < 608; i++)
            {
                if(sa_out_l00_f32[i + j * 608 + k * 608 * 608] != sa_layer_0_f32[i + j * 608 + k * 608 * 608])
                {
                    printf("layer_0_f32 mismatch: w %d, h %d, c %d, out %f, GT %f\n", i, j, k, sa_out_l00_f32[i + j * 608 + k * 608 * 608], sa_layer_0_f32[i + j * 608 + k * 608 * 608]);
                }
            }
        }
    }

    return 0;
}

void convolution_ref_c(float *p_out_f32, float *p_in_f32)
{
    
}

