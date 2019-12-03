////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Apply a low pass filter
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Applies a low pass filter to an image. Uses the k vector values defined by k_x and k_y to define the zero value.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// input_output - buffer to apply low pass to (applied in place)
/// width - width of input_output
/// height - height of input_output
/// k_max - maximum k value to allow through the filter
/// k_x - k values for x axis of input_output (size needs to equal width)
/// k_y - k values for y axis of input_output (size needs to equal height)
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void band_limit_d( __global double2* input_output, 
						   unsigned int width,
						   unsigned int height,
						   double k_max,
						   float limit_factor,
						   __global double* k_x,
						   __global double* k_y)
{
	int xid = get_global_id(0);
	int yid = get_global_id(1);

	double lim = k_max * limit_factor;

	if(xid < width && yid < height) {
		int id = xid + width*yid;
		double k = native_sqrt( k_x[xid]*k_x[xid] + k_y[yid]*k_y[yid] );
		input_output[id].x *= (k <= lim);
		input_output[id].y *= (k <= lim);
	}
}