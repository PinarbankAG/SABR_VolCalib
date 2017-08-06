#include <tbb/tbb.h>
#include <mkl.h>
#include <random>
#include <iostream>
#include <iomanip>

#define SQRT1_2 0.707106781186547524400844362105
#define PARAMETERS_COUNT 4
#define MKL_MALLOC(Size) ((double*)mkl_malloc((Size) * sizeof(double), 64))
#define NormalCDF(x) (0.5 * (1. + erf(x * SQRT1_2)))

double BSCallPrice(double F, double K, double Vol, double T, double df) {
	double d1 = (log(F / K) + (0.5 * Vol * Vol * T)) / (Vol * sqrt(T));
	double d2 = d1 - (Vol * sqrt(T));
	return df * (F * NormalCDF(d1) - K * NormalCDF(d2));
}

double InverseBSVol(double ObjCallPrice, double F, double K, double T, double DF) {
	double HiVol = 1., LoVol = 0., MidVol;
	while (HiVol - LoVol > 1.e-12) {
		MidVol = 0.5 * (HiVol + LoVol);
		if (BSCallPrice(F, K, MidVol, T, DF) > ObjCallPrice) {
			HiVol = MidVol;
		}
		else {
			LoVol = MidVol;
		}
	}
	return MidVol;
}

class SABRCalib {
private:

	// Model parameters
	// Alpha, Beta, Rho, S0.
	double Params[PARAMETERS_COUNT]; 

	// Calibration parameters
	int TimeSteps; 
	int MCIterations;

	// Input parameters
	int CallsCount;
	double Maturity;
	double DiscountFactor;
	double F0;
	double* Strikes; 
	double* BSVol; 

	// Internal Data
	bool UseControlVariates;
	int GridSize; 
	double deltaT; 
	double* MemChunk;
	double* W1; 
	double* W2_uncorrelated;
	double* W1_cumulative; 
	double* W2_cumulative; 
	double* SABRPrices; 
	double* CVAdjustment; 
	double* Jacobian; 
	double* LeftJacobian; 
	double* BSPrices; 

private:
	void ComputeWienerProcesses();
	void ComputeSABRPrices(const double* _Params, double* _SABRPrices);
	void ControlVariatesAdjustment();
	void ComputeJacobian(const double* _Params, const double SolverPrecision);

public:
	void Optimize(const double SolverPrecision);
	~SABRCalib();
	SABRCalib(int _TimeSteps, int _MonteCarloIterations, double* _InitialParams, double* _Strikes, double* _BlackVolatility, double _Maturity, double _DiscountFactor, double _F0, int _CallsCount, bool CV);
	void PrintFittedVolatility() const;
};

SABRCalib::SABRCalib(int _TimeSteps, int _MonteCarloIterations, double* _InitialParams, double* _Strikes, double* _BlackVolatility, double _Maturity, double _DiscountFactor, double _F0, int _CallsCount, bool _CV)
	: TimeSteps(_TimeSteps), MCIterations(_MonteCarloIterations), Maturity(_Maturity), DiscountFactor(_DiscountFactor), F0(_F0), CallsCount(_CallsCount), UseControlVariates(_CV) {

	// Memory allocation.
	MemChunk = MKL_MALLOC(8 * MCIterations * TimeSteps + CallsCount * (5 + 2 * PARAMETERS_COUNT));
	W1 = &MemChunk[0];
	W2_uncorrelated = &MemChunk[2 * MCIterations * TimeSteps];
	W1_cumulative = &MemChunk[4 * MCIterations * TimeSteps];
	W2_cumulative = &MemChunk[6 * MCIterations * TimeSteps];
	Jacobian = &MemChunk[8 * MCIterations * TimeSteps];
	LeftJacobian = &MemChunk[8 * MCIterations * TimeSteps + PARAMETERS_COUNT * CallsCount];
	SABRPrices = &MemChunk[8 * MCIterations * TimeSteps + 2 * PARAMETERS_COUNT * CallsCount];
	CVAdjustment = &MemChunk[8 * MCIterations * TimeSteps + (2 * PARAMETERS_COUNT + 1) * CallsCount];
	BSPrices = &MemChunk[8 * MCIterations * TimeSteps + (2 * PARAMETERS_COUNT + 2) * CallsCount];
	BSVol = &MemChunk[8 * MCIterations * TimeSteps + (2 * PARAMETERS_COUNT + 3) * CallsCount];
	Strikes = &MemChunk[8 * MCIterations * TimeSteps + (2 * PARAMETERS_COUNT + 4) * CallsCount];

	memcpy(Params, _InitialParams, PARAMETERS_COUNT * sizeof(double));
	memcpy(Strikes, _Strikes, CallsCount * sizeof(double));
	memcpy(BSVol, _BlackVolatility, CallsCount * sizeof(double));
	GridSize = 2 * MCIterations * TimeSteps;
	deltaT = Maturity / TimeSteps;
	ComputeWienerProcesses();

	// Compute the theoretical Call Prices under the Black model.
	tbb::parallel_for(0, CallsCount, [&](int i) {
		BSPrices[i] = BSCallPrice(F0, Strikes[i], BSVol[i], Maturity, DiscountFactor);
	}, tbb::affinity_partitioner{});
	
	if (UseControlVariates)
		ControlVariatesAdjustment();
}

SABRCalib::~SABRCalib() {
	mkl_free(MemChunk);
}

void SABRCalib::ComputeWienerProcesses() {
	// Draw normal variables from N(0, deltaT) 
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_WH, 1292047373);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF,
		stream,
		GridSize,
		W1,
		0., sqrt(deltaT));

	// Create Antithetic Variates from drawn values
	tbb::parallel_for(0, TimeSteps, [&](int i) {
		auto Offset = 2 * MCIterations * i;
		cblas_dcopy(MCIterations, &W1[Offset + MCIterations], 1, &W2_uncorrelated[Offset], 1);
		cblas_dcopy(MCIterations, &W2_uncorrelated[Offset], 1, &W2_uncorrelated[Offset + MCIterations], 1);
		cblas_dscal(MCIterations, -1.0, &W2_uncorrelated[Offset + MCIterations], 1);
		cblas_dcopy(MCIterations, &W1[Offset], 1, &W1[Offset + MCIterations], 1);
		cblas_dscal(MCIterations, -1.0, &W1[Offset + MCIterations], 1);
	}, tbb::affinity_partitioner{});

	// Cumulative Wiener Processes
	int Count = 2 * MCIterations;
	cblas_dcopy(Count, W1, 1, W1_cumulative, 1);
	cblas_dcopy(Count, W2_uncorrelated, 1, W2_cumulative, 1);
	for (int i = 1; i < TimeSteps; ++i) {
		int Offset = Count * i;
		vdAdd(Count, 
			&W1_cumulative[Offset - Count], 
			&W1[Offset], &W1_cumulative[Offset]);
		vdAdd(Count,
			&W2_cumulative[Offset - Count],
			&W2_uncorrelated[Offset], &W2_cumulative[Offset]);
	}
	
	vslDeleteStream(&stream);
}

void SABRCalib::ControlVariatesAdjustment() {
	int Count = 2 * MCIterations;
	int Size = Count * CallsCount;
	double* _MemChunk = MKL_MALLOC(3 * Size);
	double* Fwd = &_MemChunk[0];
	double* Options = &_MemChunk[Size];
	double* Abs = &_MemChunk[2 * Size];

	// Simulate the Fwd under the Black model, for each Strike (as the BS vol depends on the strike)
	tbb::parallel_for(0, CallsCount, [&](int i) {
		int Offset = i * Count;
		double Scal = -0.5 * BSVol[i] * BSVol[i] * Maturity;
		cblas_dcopy(Count, &W2_cumulative[(TimeSteps - 1) * Count], 1, &Fwd[Offset], 1);
		cblas_dscal(Count, BSVol[i], &Fwd[Offset], 1);
		cblas_daxpy(Count, 1.0, &Scal, 0, &Fwd[Offset], 1);
	}, tbb::affinity_partitioner{});
	vdExp(Size, Fwd, Fwd);
	cblas_dscal(Size, F0, Fwd, 1);

	// We then compute the Option prices, and compare it to the theoretical values.
	tbb::parallel_for(0, CallsCount, [&](int i) {
		int Offset = i * Count;
		cblas_dcopy(Count, &Fwd[Offset], 1, &Options[Offset], 1);
		cblas_daxpy(Count, -1.0, &Strikes[i], 0, &Options[Offset], 1);
		vdAbs(Count, &Options[Offset], &Abs[Offset]);
		vdAdd(Count, &Options[Offset], &Abs[Offset], &Options[Offset]);
		cblas_dscal(Count, 0.5 * DiscountFactor, &Options[Offset], 1);
		CVAdjustment[i] = -cblas_dasum(Count, &Options[Offset], 1) / Count;
	}, tbb::affinity_partitioner{});

	vdAdd(CallsCount, CVAdjustment, BSPrices, CVAdjustment);

	// The correlation parameter is set to 1, as the two statistics are the same (Vanilla Call prices)

	mkl_free(_MemChunk);
}

void SABRCalib::ComputeSABRPrices(const double* _Params, double* _SABRPrices) {
	
	double* _MemChunk = MKL_MALLOC(2 * MCIterations * (4 * TimeSteps + 2 * CallsCount));

	double* W2 = &_MemChunk[0];
	double* MilsteinTerm = &_MemChunk[2 * MCIterations * TimeSteps];
	double* Sigma = &_MemChunk[2 * MCIterations * TimeSteps * 2];
	double* Fwd = &_MemChunk[2 * MCIterations * TimeSteps * 3];
	double* Options = &_MemChunk[2 * MCIterations * TimeSteps * 4]; 
	double* Abs = &_MemChunk[2 * MCIterations * (TimeSteps * 4 + CallsCount)];
	
	int Count = 2 * MCIterations;
	double Alpha = _Params[0];
	double Beta = _Params[1];
	double Rho = _Params[2];
	double S0 = _Params[3];

	// Correlate W1 and W2 given current Rho parameter
	cblas_dcopy(GridSize, W1, 1, W2, 1);
	cblas_dscal(GridSize, Rho, W2, 1);
	cblas_daxpy(GridSize, sqrt(1. - Rho * Rho), W2_uncorrelated, 1, W2, 1);

	// Volatility has a closed form:
	// Sigma(t) = S0 * exp(-Alpha^2/2 * t + Alpha * W1(t))
	cblas_dcopy(GridSize, W1_cumulative, 1, Sigma, 1);
	cblas_dscal(GridSize, Alpha, Sigma, 1);
	vdExp(GridSize, Sigma, Sigma);
	cblas_dscal(GridSize, S0, Sigma, 1);

	// Forward Generating (Milstein scheme)
	vdMul(GridSize, W2, W2, MilsteinTerm);
	cblas_daxpy(GridSize, -1.0, &deltaT, 0, MilsteinTerm, 1);
	
	cblas_dcopy(Count, MilsteinTerm, 1, Fwd, 1);
	// Generate for i = 0 using initial conditions
	cblas_dscal(Count, 0.5 * Beta * S0 * S0 * pow(F0, 2. * Beta - 1.), Fwd, 1);
	cblas_daxpy(Count, S0 * pow(F0, Beta), W2, 1, Fwd, 1);
	cblas_daxpy(Count, 1., &F0, 0, Fwd, 1);
	// Reflexion scheme 
	vdAbs(Count, Fwd, Fwd);

	// Then, loop until maturity
	for (int i = 1; i < TimeSteps; ++i) {
		int Offset = i * Count;
		vdMul(Count, &W2[Offset], &Sigma[Offset - Count], &Fwd[Offset]);
		vmdPowx(Count, &Fwd[Offset - Count], Beta, &W2[Offset], VML_EP);
		vdMul(Count, &Fwd[Offset], &W2[Offset], &Fwd[Offset]);
		vdAdd(Count, &Fwd[Offset], &Fwd[Offset - Count], &Fwd[Offset]);
		cblas_dscal(Count, 0.5 * Beta, &MilsteinTerm[Offset], 1);
		vdMul(Count, &MilsteinTerm[Offset], &Sigma[Offset - Count], &MilsteinTerm[Offset]);
		vdMul(Count, &MilsteinTerm[Offset], &Sigma[Offset - Count], &MilsteinTerm[Offset]);
		vmdPowx(Count, &Fwd[Offset - Count], 2. * Beta - 1., &W2[Offset], VML_EP);
		vdMul(Count, &MilsteinTerm[Offset], &W2[Offset], &MilsteinTerm[Offset]);
		vdAdd(Count, &MilsteinTerm[Offset], &Fwd[Offset], &Fwd[Offset]);
		vdAbs(Count, &Fwd[Offset], &Fwd[Offset]);
	}

	// The last Row of Fwd contains the simulated forwards at expiry.
	tbb::parallel_for(0, CallsCount, [&](int i) {
		int Offset = i * Count;
		cblas_dcopy(Count, &Fwd[Count * (TimeSteps - 1)], 1, &Options[Offset], 1);
		cblas_daxpy(Count, -1.0, &Strikes[i], 0, &Options[Offset], 1);
		vdAbs(Count, &Options[Offset], &Abs[Offset]);
		vdAdd(Count, &Options[Offset], &Abs[Offset], &Options[Offset]);
		cblas_dscal(Count, 0.5, &Options[Offset], 1);
		_SABRPrices[i] = cblas_dasum(Count, &Options[Offset], 1) / Count;
	}, tbb::affinity_partitioner{});

	// Add the Control Variates correction term, after discounting.
	cblas_dscal(CallsCount, DiscountFactor, _SABRPrices, 1);
	if (UseControlVariates)
		vdAdd(CallsCount, CVAdjustment, _SABRPrices, _SABRPrices);

	tbb::parallel_for(0, CallsCount, [&](int i) {
		_SABRPrices[i] = InverseBSVol(_SABRPrices[i], F0, Strikes[i], Maturity, DiscountFactor);
	}, tbb::affinity_partitioner{});

	vdSub(CallsCount, _SABRPrices, BSVol, _SABRPrices);

	mkl_free(_MemChunk);
}

void SABRCalib::ComputeJacobian(const double* _Params, const double SolverPrecision) {	
	// Change one parameter at a time, and compute the Jacobian using Central Differences (ie f'(x) = (f(x+Eps)-f(x-Eps))/(2*Eps)
	tbb::parallel_for(0, PARAMETERS_COUNT, [&](int i) {
		double LeftParams[PARAMETERS_COUNT];
		double RightParams[PARAMETERS_COUNT];
		memcpy(LeftParams, _Params, PARAMETERS_COUNT * sizeof(double));
		memcpy(RightParams, _Params, PARAMETERS_COUNT * sizeof(double));
		LeftParams[i] -= SolverPrecision;
		RightParams[i] += SolverPrecision;
		ComputeSABRPrices(LeftParams, &LeftJacobian[i * CallsCount]);
		ComputeSABRPrices(RightParams, &Jacobian[i * CallsCount]);
		memcpy(&Jacobian[i * CallsCount], SABRPrices, CallsCount * sizeof(double));
	}, tbb::affinity_partitioner{});
	vdSub(CallsCount * PARAMETERS_COUNT, Jacobian, LeftJacobian, Jacobian);
	cblas_dscal(CallsCount * PARAMETERS_COUNT, 0.5 / SolverPrecision, Jacobian, 1);
}

/* Non Linear Least Squares Optimizer */
void SABRCalib::Optimize(const double SolverPrecision) {
	/* Precisions for stop-criteria. */
	double SolverEpsilon[6] = { SolverPrecision,SolverPrecision,SolverPrecision,SolverPrecision,SolverPrecision,SolverPrecision };
	double TrustRegionArea = 1.0;
	auto INF = DBL_MAX;
	double LowerBounds[PARAMETERS_COUNT] = { 0., 0., -1., 0. };
	double UpperBounds[PARAMETERS_COUNT] = { INF, 1., 1., INF };
	int MaxIterations = 15;
	int TrialStepIterations = 15;

	/* Initialize solver (allocate memory, set initial values, ...) */
	int RCI_Request = 0;
	_TRNSP_HANDLE_t handle;
	int info[6];
	memset(SABRPrices, 0, CallsCount * sizeof(double));
	memset(Jacobian, 0, CallsCount * PARAMETERS_COUNT * sizeof(double));
	auto ParamsCnt = PARAMETERS_COUNT;
	if (dtrnlspbc_init(&handle, &ParamsCnt, &CallsCount, Params, LowerBounds, UpperBounds, SolverEpsilon, &MaxIterations, &TrialStepIterations, &TrustRegionArea) != TR_SUCCESS) {
		std::cout << "Error in Solver initialization.\n";
		MKL_Free_Buffers();
		return;
	}

	/* Checks the correctness of handle and arrays containing Jacobian matrix,
	objective function, lower and upper bounds, and stopping criteria. */
	if (dtrnlspbc_check(&handle, &ParamsCnt, &CallsCount, Jacobian, SABRPrices, LowerBounds, UpperBounds, SolverEpsilon, info) != TR_SUCCESS) {
		std::cout << "Please check the input parameters.\n";
		MKL_Free_Buffers();
		return;
	}

	/* RCI cycle. */
	do
	{
		if (dtrnlspbc_solve(&handle, SABRPrices, Jacobian, &RCI_Request) != TR_SUCCESS) {
			std::cout << "Error in Solver.\n";
			MKL_Free_Buffers();
			return;
		}
		if (RCI_Request == 1) {
			ComputeSABRPrices(Params, SABRPrices);
		}
		if (RCI_Request == 2) {
			ComputeJacobian(Params, SolverPrecision);
		}
	} while (RCI_Request > -1 || RCI_Request < -6);
	dtrnlspbc_delete(&handle);
	MKL_Free_Buffers();
}

void SABRCalib::PrintFittedVolatility() const {
	std::setprecision(7);
	std::cout << "Strike\tB-S\tSABR\n";
	for (int i = 0; i < CallsCount; ++i) {
		std::cout << Strikes[i] << '\t' << 100. * BSVol[i] << '\t' << 100. * (BSVol[i] + SABRPrices[i]) << "\n";
	}
	std::cout << "Alpha :\t" << Params[0] << "\tBeta :\t" << Params[1] << "\tRho :\t" << Params[2] << "\tS0 :\t" << Params[3] * exp(Params[0] * Params[0] * Maturity / 2.) << "\t";
	auto rmse = 100. * cblas_dnrm2(CallsCount, SABRPrices, 1) / CallsCount;
	std::cout << "ImpVol RMSE :\t" << rmse << "\n";
}	

int main() {
	// 2Y Data
	int TimeSteps = 768;
	int MCIterations = 3336;
	// Alpha, Beta, Rho, S0
	double InitialParams[4] = {
		0.393654,
		0.786748,
		-0.57492,
		1.14185};
	double Strikes[] = { 
		2493.66,	2992.39,	3491.12,	3740.48,	3989.85,	
		4239.21,	4488.58,	4737.94,	4862.63,	4987.31,	
		5111.99,	5236.68,	5486.04,	5735.41,	5984.77,	
		6234.14,	6483.50,	6982.23,	7480.97};
	double BSVol[] = { 
		.286,	.249,	.2199,	.2084,	.1984,
		.1895,	.1815,	.1744,	.171,	.1679,
		.165,	.1623,	.1576,	.1541,	.1516,
		.1501,	.1494,	.1503,	.1543 };
	double Maturity = 2.0;
	double DiscountFactor = exp(0.1 * 2. / 100.);
	double F0 = 4686.14;
	int CallsCount = 19;
	bool useControlVariates = true;
	SABRCalib Calib(TimeSteps, MCIterations, InitialParams, Strikes, BSVol, Maturity, DiscountFactor, F0, CallsCount, useControlVariates);
	Calib.Optimize(1e-9);
	Calib.PrintFittedVolatility();
	system("pause");
	return 0;
}
