/*******************************************************************************
* Copyright (c) 2024 Cadence Design Systems, Inc.
* 
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
* 
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/
#ifndef __XA_NNLIB_SCRATCH_SIZE_API_H__
#define __XA_NNLIB_SCRATCH_SIZE_API_H__


#if defined(__cplusplus)
	extern "C"
{
#endif

/* HiFi5 getsize APIs */
WORD32 xa_nn_transpose_conv_getsize_hifi5(
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 output_height,
    WORD32 output_width,
    WORD32 output_channels,
    WORD32 num_groups,
    WORD32 kernel_precision,
    WORD32 output_precision);

WORD32 xa_nn_conv2d_std_getsize_hifi5(
		WORD32 input_height,
		WORD32 input_width,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 kernel_channels,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 x_stride,
		WORD32 x_padding,
		WORD32 out_height,
		WORD32 out_width,
		WORD32 output_channels,
		WORD32 input_precision,
		WORD32 kernel_precision,
		WORD32 dilation_height,
		WORD32 dilation_width,
		WORD32 out_data_format);

WORD32 xa_nn_conv2d_getsize_hifi5(
		WORD32 input_height,
		WORD32 input_width,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 kernel_channels,
		WORD32 dilation_height,
		WORD32 dilation_width,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 x_stride,
		WORD32 x_padding,
		WORD32 out_height,
		WORD32 out_width,
		WORD32 output_channels,
		WORD32 input_precision,
		WORD32 kernel_precision,
		WORD32 out_data_format
		);

WORD32 xa_nn_conv2d_std_getsize_sym4s_hifi5(
		WORD32 input_height,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 out_height,
		WORD32 out_channels,
		WORD32 input_precision);

WORD32 xa_nn_dilated_conv2d_std_getsize_hifi5(
		WORD32 input_height,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 out_height,
		WORD32 out_channels,
		WORD32 input_precision,
		WORD32 dilation_height
		);

WORD32 xa_nn_conv2d_depthwise_getsize_hifi5
	(WORD32 input_height
	 ,WORD32 input_width
	 ,WORD32 input_channels
	 ,WORD32 kernel_height
	 ,WORD32 kernel_width
	 ,WORD32 channels_multiplier
	 ,WORD32 x_stride
	 ,WORD32 y_stride
	 ,WORD32 x_padding
	 ,WORD32 y_padding
	 ,WORD32 output_height
	 ,WORD32 output_width
	 ,WORD32 circ_buf_precision
	 ,WORD32 inp_data_format
	);

WORD32 xa_nn_dilated_conv2d_depthwise_getsize_hifi5
	(WORD32 input_height
	 ,WORD32 input_width
	 ,WORD32 input_channels
	 ,WORD32 kernel_height
	 ,WORD32 kernel_width
	 ,WORD32 channels_multiplier
     ,WORD32 dilation_height
     ,WORD32 dilation_width
	 ,WORD32 x_stride
	 ,WORD32 y_stride
	 ,WORD32 x_padding
	 ,WORD32 y_padding
	 ,WORD32 output_height
	 ,WORD32 output_width
	 ,WORD32 circ_buf_precision
	 ,WORD32 inp_data_format
	);

WORD32 xa_nn_avgpool_getsize_hifi5(
        WORD32 input_channels,
        WORD32 inp_precision,
        WORD32 out_precision,
        WORD32 input_height,
        WORD32 input_width,
        WORD32 kernel_height,
        WORD32 kernel_width,
        WORD32 x_stride,
        WORD32 y_stride,
        WORD32 x_padding,
        WORD32 y_padding,
        WORD32 out_height,
        WORD32 out_width,
        WORD32 inp_data_format,
        WORD32 out_data_format);

WORD32 xa_nn_maxpool_getsize_hifi5(
		WORD32 input_channels,
		WORD32 inp_precision,
		WORD32 out_precision,
		WORD32 input_height,
		WORD32 input_width,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 x_stride,
		WORD32 y_stride,
		WORD32 x_padding,
		WORD32 y_padding,
		WORD32 out_height,
		WORD32 out_width,
		WORD32 inp_data_format,
		WORD32 out_data_format);

WORD32 xa_nn_batch_matmul_getsize_hifi5(
        const WORD32 *const p_mat1_shape,
        const WORD32 *const p_mat2_shape,
        WORD32 mat1_transpose,
        WORD32 mat2_transpose,
        WORD32 mat1_precision,
        WORD32 mat2_precision);

int get_softmax_scratch_size_hifi5(int inp_precision, int out_precision, int length);

WORD32 xa_nn_reduce_getsize_nhwc_hifi5(WORD32  inp_precision
		,const WORD32 *const p_inp_shape
		,WORD32 num_inp_dims
		,const WORD32 *p_axis
		,WORD32 num_axis_dims
		,WORD32 reduce_ops);

WORD32 xa_nn_lstm_getsize_hifi5(
        WORD32 n_batch,
        WORD32 n_itr,
        WORD32 n_cell,
        WORD32 cell_state_precision);

/* HiFi4 getsize APIs */
WORD32 xa_nn_transpose_conv_getsize_hifi4(
    WORD32 input_height,
    WORD32 input_width,
    WORD32 input_channels,
    WORD32 kernel_height,
    WORD32 kernel_width,
    WORD32 x_stride,
    WORD32 y_stride,
    WORD32 output_height,
    WORD32 output_width,
    WORD32 output_channels,
    WORD32 num_groups,
    WORD32 kernel_precision,
    WORD32 output_precision);

WORD32 xa_nn_conv2d_std_getsize_hifi4(
		WORD32 input_height,
		WORD32 input_width,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 kernel_channels,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 x_stride,
		WORD32 x_padding,
		WORD32 out_height,
		WORD32 out_width,
		WORD32 output_channels,
		WORD32 input_precision,
		WORD32 kernel_precision,
		WORD32 dilation_height,
		WORD32 dilation_width,
		WORD32 out_data_format);

WORD32 xa_nn_conv2d_getsize_hifi4(
		WORD32 input_height,
		WORD32 input_width,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 kernel_channels,
		WORD32 dilation_height,
		WORD32 dilation_width,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 x_stride,
		WORD32 x_padding,
		WORD32 out_height,
		WORD32 out_width,
		WORD32 output_channels,
		WORD32 input_precision,
		WORD32 kernel_precision,
		WORD32 out_data_format
		);

WORD32 xa_nn_conv2d_std_getsize_sym4s_hifi4(
		WORD32 input_height,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 out_height,
		WORD32 out_channels,
		WORD32 input_precision);

WORD32 xa_nn_dilated_conv2d_std_getsize_hifi4(
		WORD32 input_height,
		WORD32 input_channels,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 y_stride,
		WORD32 y_padding,
		WORD32 out_height,
		WORD32 out_channels,
		WORD32 input_precision,
		WORD32 dilation_height
		);

WORD32 xa_nn_conv2d_depthwise_getsize_hifi4
	(WORD32 input_height
	 ,WORD32 input_width
	 ,WORD32 input_channels
	 ,WORD32 kernel_height
	 ,WORD32 kernel_width
	 ,WORD32 channels_multiplier
	 ,WORD32 x_stride
	 ,WORD32 y_stride
	 ,WORD32 x_padding
	 ,WORD32 y_padding
	 ,WORD32 output_height
	 ,WORD32 output_width
	 ,WORD32 circ_buf_precision
	 ,WORD32 inp_data_format
	);

WORD32 xa_nn_dilated_conv2d_depthwise_getsize_hifi4
	(WORD32 input_height
	 ,WORD32 input_width
	 ,WORD32 input_channels
	 ,WORD32 kernel_height
	 ,WORD32 kernel_width
	 ,WORD32 channels_multiplier
     ,WORD32 dilation_height
     ,WORD32 dilation_width
	 ,WORD32 x_stride
	 ,WORD32 y_stride
	 ,WORD32 x_padding
	 ,WORD32 y_padding
	 ,WORD32 output_height
	 ,WORD32 output_width
	 ,WORD32 circ_buf_precision
	 ,WORD32 inp_data_format
	);

WORD32 xa_nn_avgpool_getsize_hifi4(
        WORD32 input_channels,
        WORD32 inp_precision,
        WORD32 out_precision,
        WORD32 input_height,
        WORD32 input_width,
        WORD32 kernel_height,
        WORD32 kernel_width,
        WORD32 x_stride,
        WORD32 y_stride,
        WORD32 x_padding,
        WORD32 y_padding,
        WORD32 out_height,
        WORD32 out_width,
        WORD32 inp_data_format,
        WORD32 out_data_format);

WORD32 xa_nn_maxpool_getsize_hifi4(
		WORD32 input_channels,
		WORD32 inp_precision,
		WORD32 out_precision,
		WORD32 input_height,
		WORD32 input_width,
		WORD32 kernel_height,
		WORD32 kernel_width,
		WORD32 x_stride,
		WORD32 y_stride,
		WORD32 x_padding,
		WORD32 y_padding,
		WORD32 out_height,
		WORD32 out_width,
		WORD32 inp_data_format,
		WORD32 out_data_format);

WORD32 xa_nn_batch_matmul_getsize_hifi4(
        const WORD32 *const p_mat1_shape,
        const WORD32 *const p_mat2_shape,
        WORD32 mat1_transpose,
        WORD32 mat2_transpose,
        WORD32 mat1_precision,
        WORD32 mat2_precision);

int get_softmax_scratch_size_hifi4(int inp_precision, int out_precision, int length);

WORD32 xa_nn_reduce_getsize_nhwc_hifi4(WORD32  inp_precision
		,const WORD32 *const p_inp_shape
		,WORD32 num_inp_dims
		,const WORD32 *p_axis
		,WORD32 num_axis_dims
		,WORD32 reduce_ops);

WORD32 xa_nn_lstm_getsize_hifi4(
        WORD32 n_batch,
        WORD32 n_itr,
        WORD32 n_cell,
        WORD32 cell_state_precision);

#if defined(__cplusplus)
}
#endif
#endif /* __XA_NNLIB_SCRATCH_SIZE_API_H__ */
