#include <string>
#include <cstdio> //sscanf
#include <iostream>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/cont/DataSet.h>
#include <vtkm/cont/DataSetBuilderUniform.h>
#include <vtkm/cont/DataSetFieldAdd.h>
#include <vtkm/io/writer/VTKDataSetWriter.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/DispatcherMapField.h>
#include<vtkm/VectorAnalysis.h>
// Unused (but may be useful later)
//#include <vtkm/filter/ResultDataSet.h>
//#include <boost/math/tools/roots.hpp>

//We first check if VTKM_DEVICE_ADAPTER is defined, so that when TBB and CUDA
//includes this file we use the device adapter that they have set.
#ifndef VTKM_DEVICE_ADAPTER
#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_SERIAL
#endif

// Resolution: N_X x N_Y
#define N_X 30
#define N_Y 30
/* VTK-m notes:
 * All indices should be the vtkm::Id type
 * FloatDefault can be changed with
 * #define VTKM_USE_DOUBLE_PRECISION or  #define VTKM_NO_DOUBLE_PRECISION
 */

// Define some types for the Bézier curves. These may become templated.
// A cubic Bézier curve is defined by four control points
typedef typename vtkm::Vec<vtkm::FloatDefault, 2> point_t;
typedef typename vtkm::FloatDefault in_bezier_t; // Inside Bézier type
typedef typename vtkm::Vec<vtkm::FloatDefault, 2> cp_t;
typedef typename vtkm::Vec<cp_t, 4> bezier_t;

// A Brush is a functor defined from [0,1] -> [0,\infty]
// Returns the width of the brush at that point along the curve
class Brush
{
public:
	vtkm::FloatDefault operator()(vtkm::FloatDefault t) const
	{
		return 0.07;
	}
};

std::vector<bezier_t> read_control_point_file(std::string filename)
{
	std::ifstream in(filename);
    std::string line;
    std::cout << "Reading from " << filename << std::endl;
	vtkm::FloatDefault p0u, p0v, p1u, p1v, p2u, p2v, p3u, p3v;
	std::vector<bezier_t> control_points;
    while(std::getline(in, line)) {
		if (line.length() < 12)
			continue;
		sscanf(line.c_str(), "(%f,%f) (%f,%f) (%f,%f) (%f,%f)",
				&p0u,&p0v, &p1u,&p1v, &p2u,&p2v, &p3u,&p3v);
		control_points.push_back(bezier_t(
				cp_t(p0u, p0v),
				cp_t(p1u, p1v),
				cp_t(p2u, p2v),
				cp_t(p3u, p3v)));
	}
	// TEST
	for (int cpi = 0; cpi < control_points.size(); ++cpi) {
		for (int cpj = 0; cpj < 4; ++cpj) {
			std::cout << cpi << ": " << control_points[cpi][cpj][0] << "," <<
					control_points[cpi][cpj][1] << std::endl;
		}
	}
	return control_points;
}


/* Define a VTK-m worklet */
namespace vtkm {
namespace worklet {

class CubicBezier
{
public:
	typedef typename std::vector<bezier_t> AllBezierCurves;
	// Evaluate a Bézier curve at some point t \in [0,1]
	static inline point_t eval(bezier_t p, vtkm::FloatDefault t)
	{
		if (t < 0.f || t > 1.f)
			throw vtkm::cont::ErrorExecution("t must be between 0 and 1");
		vtkm::FloatDefault t2, t3, u, v;	
		t2 = t * t;
		t3 = t2 * t;
		u = p[0][0] +
			3*(p[1][0] - p[0][0])*t +
			3*(p[0][0] - 2*p[1][0] + p[2][0])*t2 +
			(3*p[1][0] - p[0][0] - 3*p[2][0] + p[3][0])*t3;
		v = p[0][1] +
			3*(p[1][1] - p[0][1])*t +
			3*(p[0][1] - 2*p[1][1] + p[2][1])*t2 +
			(3*p[1][1] - p[0][1] - 3*p[2][1] + p[3][1])*t3;
		return point_t(u,v);
	}

	// Functor to return F(t) = dB_udt*(B_u - u) + dB_vdt*(B_v - v)
	// and its derivative, F'(t) = dF / dt
	// Evaluated at a point (u,v)
	// i.e. the polynomial of which we want to find the roots
	template<class T>
	struct bezier_poly_func
	{
		T u, v, p0u, p1u, p2u, p3u, p0v, p1v, p2v, p3v;
		// cp gets renamed to match the symbolic coefficients output by Matlab
		bezier_poly_func(bezier_t const& cp, point_t point)
		{
			u = point[0]; v = point[1];
			p0u = cp[0][0]; p0v = cp[0][1];
			p1u = cp[1][0]; p1v = cp[1][1];
			p2u = cp[2][0]; p2v = cp[2][1];
			p3u = cp[3][0]; p3v = cp[3][1];
		}
		std::tuple<T, T> operator()(T const& t)
		{
#define sq(x) ((x)*(x))
			T t2 = t * t;
			T t3 = t2 * t;
			T t4 = t3 * t;
			T t5 = t4 * t;
			T Ft = t5 * ((p0u - 3*p1u + 3*p2u - p3u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (p0v - 3*p1v + 3*p2v - p3v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)) +
					t4 * (- (6*p0u - 12*p1u + 6*p2u)*(p0u - 3*p1u + 3*p2u - p3u) - (6*p0v - 12*p1v + 6*p2v)*(p0v - 3*p1v + 3*p2v - p3v) - (3*p0u - 6*p1u + 3*p2u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (3*p0v - 6*p1v + 3*p2v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)) +
					t3 * ((3*p0u - 3*p1u)*(p0u - 3*p1u + 3*p2u - p3u) + (3*p0v - 3*p1v)*(p0v - 3*p1v + 3*p2v - p3v) + (3*p0u - 3*p1u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (3*p0u - 6*p1u + 3*p2u)*(6*p0u - 12*p1u + 6*p2u) + (3*p0v - 3*p1v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v) + (3*p0v - 6*p1v + 3*p2v)*(6*p0v - 12*p1v + 6*p2v)) +
					t2 * (- (3*p0u - 3*p1u)*(3*p0u - 6*p1u + 3*p2u) - (3*p0u - 3*p1u)*(6*p0u - 12*p1u + 6*p2u) - (3*p0v - 3*p1v)*(3*p0v - 6*p1v + 3*p2v) - (3*p0v - 3*p1v)*(6*p0v - 12*p1v + 6*p2v) - (p0u - u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (p0v - v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)) +
					t * (sq(3*p0u - 3*p1u) + sq(3*p0v - 3*p1v) + (p0u - u)*(6*p0u - 12*p1u + 6*p2u) + (p0v - v)*(6*p0v - 12*p1v + 6*p2v)) +
			 		 - (p0u - u)*(3*p0u - 3*p1u) - (p0v - v)*(3*p0v - 3*p1v);

			T dFdt = 5 * t4 * ((p0u - 3*p1u + 3*p2u - p3u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (p0v - 3*p1v + 3*p2v - p3v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)) +
					4 * t3 * (- (6*p0u - 12*p1u + 6*p2u)*(p0u - 3*p1u + 3*p2u - p3u) - (6*p0v - 12*p1v + 6*p2v)*(p0v - 3*p1v + 3*p2v - p3v) - (3*p0u - 6*p1u + 3*p2u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (3*p0v - 6*p1v + 3*p2v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)) +
					3 * t2 * ((3*p0u - 3*p1u)*(p0u - 3*p1u + 3*p2u - p3u) + (3*p0v - 3*p1v)*(p0v - 3*p1v + 3*p2v - p3v) + (3*p0u - 3*p1u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) + (3*p0u - 6*p1u + 3*p2u)*(6*p0u - 12*p1u + 6*p2u) + (3*p0v - 3*p1v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v) + (3*p0v - 6*p1v + 3*p2v)*(6*p0v - 12*p1v + 6*p2v)) +
					2 * t * (- (3*p0u - 3*p1u)*(3*p0u - 6*p1u + 3*p2u) - (3*p0u - 3*p1u)*(6*p0u - 12*p1u + 6*p2u) - (3*p0v - 3*p1v)*(3*p0v - 6*p1v + 3*p2v) - (3*p0v - 3*p1v)*(6*p0v - 12*p1v + 6*p2v) - (p0u - u)*(3*p0u - 9*p1u + 9*p2u - 3*p3u) - (p0v - v)*(3*p0v - 9*p1v + 9*p2v - 3*p3v)) +
					(sq(3*p0u - 3*p1u) + sq(3*p0v - 3*p1v) + (p0u - u)*(6*p0u - 12*p1u + 6*p2u) + (p0v - v)*(6*p0v - 12*p1v + 6*p2v));
			return std::make_tuple(Ft, dFdt);
		}
	};


	// Check whether each point is within t units of the Bezier curve
	class CheckIfInsideBezierCurves : public vtkm::worklet::WorkletMapField
	{
	public:
		typedef void ControlSignature(
				FieldIn<> point, 
				FieldOut<> in_or_out_field);
		typedef void ExecutionSignature(_1, _2);
		typedef _1 InputDomain; // Parallelize over the points

		Brush brush;
		AllBezierCurves curve_array;
		const vtkm::FloatDefault start = 0.5;
		const vtkm::FloatDefault step = 0.25;
		int octaves = 9; // Accurate to 2^(octaves-1) pixels

		VTKM_CONT
		CheckIfInsideBezierCurves(
				Brush brush_arg, AllBezierCurves curve_array_arg) :
				brush(brush_arg), curve_array(curve_array_arg) {}

		VTKM_EXEC
		void operator()(const point_t &point, in_bezier_t &in_or_out) const
		{
			vtkm::FloatDefault diffp, diffm;
			vtkm::FloatDefault t;
			vtkm::FloatDefault s;
			bezier_t B;
			for (vtkm::Id bi = 0; bi < curve_array.size(); ++bi) {
				t = start;
				s = step;
				B = curve_array[bi];
				for (int o = 0; o < octaves; ++o) {
					diffp = vtkm::MagnitudeSquared(eval(B, t + s) - point);
					diffm = vtkm::MagnitudeSquared(eval(B, t - s) - point);
					if (diffp < diffm)
						t += s;
					else
						t -= s;
					s /= 2;
				}
				if (vtkm::Magnitude(point - eval(B, t)) < brush(t)) {
					in_or_out = 1.0f;
				}
				// else assume in_or_out is zeroed
			}
		}
	};

	// Prints a summary of the sample points
	void PrintSamplePoints(
			vtkm::cont::ArrayHandle<point_t> sample_points,
			vtkm::Id2 dim)
	{
		vtkm::Id sumlen = 4;
		typedef typename vtkm::cont::internal::Storage<
				point_t, VTKM_DEFAULT_STORAGE_TAG> StorageType;
		StorageType::PortalConstType readPortal =
				sample_points.GetPortalConstControl();
		// We don't need this because there's dim passed in
		// vtkm::Id numElements = readPortal.GetNumberOfValues();
		std::cout << "sample_points, dim(" << dim[0] << "," << dim[1]
				<< ")" << std::endl;
		for (vtkm::Id xx = 0; xx < dim[0]; ++xx) {
			if (xx > sumlen && xx < dim[0] - 2) {
				if (xx == sumlen+1) std::cout <<
						"."<<std::endl<<"."<<std::endl<<"."<<std::endl;
				continue;
			}
			for (vtkm::Id yy = 0; yy < dim[1]; ++yy) {
				if (yy > sumlen && yy < dim[1] - 2) {
					if (yy == sumlen+1)
						std::cout << "... ";
					continue;
				}
				std::cout << readPortal.Get(xx * dim[0] + yy) << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	// Print a 1 if the point is inside the bezier curve, 0 if not.
	// Note: dim3[2] better be 1
	void PrintInBezier(
			vtkm::cont::ArrayHandle<in_bezier_t> in_or_out,
			vtkm::Id2 dim)
	{
		typedef typename vtkm::cont::internal::Storage<
				in_bezier_t, VTKM_DEFAULT_STORAGE_TAG> StorageType;
		StorageType::PortalConstType readPortal =
				in_or_out.GetPortalConstControl();
		// We don't need this because there's dim passed in
		// vtkm::Id numElements = readPortal.GetNumberOfValues();
		std::cout << "in_or_out, dim(" << dim[0] << "," << dim[1]
				<< ")" << std::endl;
		for (vtkm::Id xx = 0; xx < dim[0]; ++xx) {
			for (vtkm::Id yy = 0; yy < dim[1]; ++yy) {
				std::cout << ((int) readPortal.Get(xx * dim[0] + yy));
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	vtkm::cont::ArrayHandle<in_bezier_t> Run(
			vtkm::cont::ArrayHandle<point_t> sample_point_array,
			AllBezierCurves control_points,
			vtkm::Id2 dim)
	{
		// Initialize the output array of the worklet
		vtkm::cont::ArrayHandle<in_bezier_t> result_array;
		//result_array.Allocate(dim[0] * dim[1]); // May not be needed?
		// Initialize the worklet and its dispatcher
		Brush brush;
		CheckIfInsideBezierCurves bezier_worklet(brush, control_points);
		vtkm::worklet::DispatcherMapField<CheckIfInsideBezierCurves>
				bezier_dispatcher(bezier_worklet);
		bezier_dispatcher.Invoke(sample_point_array, result_array);
		PrintSamplePoints(sample_point_array, dim);
		PrintInBezier(result_array, dim);
		return(result_array);
	}
}; // end CubicBezier

} // end namespace worklet
} // end namespace vtkm

int main(int argc, char **argv)
{
	const int n_points = N_X * N_Y;
	vtkm::Id2 dim = vtkm::Id2(N_X, N_Y);

	// Start with this Bézier curve -- it makes a c shape
	// std::vector<bezier_t> control_points = std::vector<bezier_t>();
	// control_points.push_back(bezier_t(
	// 		cp_t(0.7, 0.6),
	// 		cp_t(0.3, 1.0),
	// 		cp_t(0.3, 0.0),
	// 		cp_t(0.7, 0.4)));
	std::vector<bezier_t> control_points = read_control_point_file("cs.txt");

	// TODO: This should probably be a function
	// generate the sample_points
	// O ----- x (row-major)
	// |
	// |
	// |
	// v
	// y
	std::vector<point_t> sample_points = std::vector<point_t>(n_points);
	vtkm::FloatDefault spacing_x = 1.f / (N_X - 1);
	vtkm::FloatDefault spacing_y = 1.f / (N_Y - 1);
	vtkm::FloatDefault x_val = 0.0f; // Origin
	vtkm::FloatDefault y_val = 0.0f; // Origin
	for (int row = 0; row < N_Y; ++row) {
		x_val = 0.0f;
		for (int col = 0; col < N_X; ++col) {
			sample_points[row * N_Y + col] = point_t(x_val, y_val);
			x_val += spacing_x;
		}
		y_val += spacing_y;
	}
	vtkm::cont::ArrayHandle<point_t> sample_point_array =
			vtkm::cont::make_ArrayHandle(sample_points);
	// Generate a dataset which matches these points exactly
	vtkm::cont::DataSetBuilderUniform dataset_builder;
	vtkm::cont::DataSet dataset = dataset_builder.Create(
			dim,
			vtkm::Vec<vtkm::FloatDefault, 2>(0.f, 0.f),         //Custom Origin
			vtkm::Vec<vtkm::FloatDefault, 2>(1.f/N_X, 1.f/N_Y));//Custom spacing
	// NOTE: dynamic cast isn't working
	//auto dyn_point_array = dataset.GetCoordinateSystem().GetData();
	//dyn_point_array.PrintSummary(std::cout);
	//
	//vtkm::cont::ArrayHandle<point_t> sample_point_array =
	//		dyn_point_array.Cast<vtkm::cont::ArrayHandle<point_t> >();
	// Add field
	//vtkm::cont::Field in_or_out_field("in_or_out_field",
	//			vtkm::cont::Field::ASSOC_POINTS, in_or_out, n_points);
	//dataset.AddField(in_or_out_field);
	//vtkm::cont::ArrayHandleUniformPointCoordinates sample_point_array(
	//		vtkm::Id3(N_X, N_Y, 1),
	//		vtkm::make_Vec<vtkm::FloatDefault>(0.f, 0.f, 0.f),
	//		vtkm::make_Vec<vtkm::FloatDefault>(1.f/N_X, 1.f/N_Y, 1.f));

	// Eventually, do this with 2D
	// Create the points at which the Bézier curves will be checked
	// Use (0,0) (it's also the default origin) and custom spacing
	//vtkm::cont::ArrayHandleUniformPointCoordinates2D sample_points(
	//		dim,
	//		vtkm::make_Vec<vtkm::FloatDefault>(0.f, 0.f),
	//		vtkm::make_Vec<vtkm::FloatDefault>(1.f/N_X, 1.f/N_Y));
	
	// The input -> output model here matches what we're trying to do.
	vtkm::worklet::CubicBezier bezier;
	vtkm::cont::ArrayHandle<in_bezier_t> result_array = bezier.Run(
			sample_point_array,
			control_points,
			dim);

	// Add the result to a field in a dataset
	vtkm::cont::DataSetFieldAdd dataSetFieldAdd;
	dataSetFieldAdd.AddCellField(dataset, "in_bezier", result_array);
	
	//if (!result_dataset.IsValid()) {
	//	std::cout << "Dataset is invalid" << std::endl;
	//	exit(1);
	//}

	// Write the output
	const std::string out_file = "bezier.vtk";
	std::cout << "Writing file: " << out_file << std::endl;
	vtkm::io::writer::VTKDataSetWriter writer(out_file);
	writer.WriteDataSet(dataset);

	return 0;
}

