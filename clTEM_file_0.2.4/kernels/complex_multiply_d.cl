////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Multiply two complex images
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Multiples two complex images element wise
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// input_a - first image to be multiplied
/// input_b - second image to be multiplied
/// output - output of multiplication
/// width - width of hte inputs
/// height - height of the outputs
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void complex_multiply_d( __global double2* input_a,
								 __global double2* input_b,
								 __global double2* output, 
								 unsigned int width,
								 unsigned int height)
{
	int xid = get_global_id(0);
	int yid = get_global_id(1);
	if (xid < width && yid < height) {
		int id = xid + width * yid;
		output[id].x = input_a[id].x * input_b[id].x - input_a[id].y * input_b[id].y;
		output[id].y = input_a[id].x * input_b[id].y + input_a[id].y * input_b[id].x;
	}
}