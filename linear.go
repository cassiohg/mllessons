package main

import (
	"fmt"
	"time"
	"math/rand"
	"math"
)

type float = float64

// function to return random float numbers between -1 and 1.
func random () float {
	return float(rand.Float32()*2) -1
}

// structure representing a data point.
type Point struct {
	y 	      float64   // point classification.
	x 	      []float64 // slice/array of data point parameters.
	originalX []float64 // slice reference to original backing array for 'x' parameters.
}

/* a method that receives a transformation function as argument that is expected to receive a slice of floats and return 
another slice of floats (they don't have to have the same lengths). */
func (point *Point) transform (transformation func([]float64) []float64) {
	// substitute reference to slice 'x' of a point by a transformation applied to it's own original 'x' slice.
	point.x = transformation(point.originalX)
}

/* sets given 'target' Point object that uses the same backing array for attribute 'originalX' slice and a copy of attribute 
'x' with brand new slice and backing array (full copy). Attribute 'y' is copied by value to returned object. */
func (point *Point) weakCopyTo (target *Point) {
	target.y = point.y
	target.originalX = point.originalX
	target.x = make([]float64, len(point.x)) // making brand new backing array for 'x'.
	for i := range point.x { target.x[i] = point.x[i]} // copying values from 'point's x to 'newpoint's 'x'.
}

/* returns another Point object that uses the same backing array for attribute 'originalX' slice and a copy of attribute 
'x' with brand new slice and backing array (full copy). Attribute 'y' is copied by value to returned object. */
func (point *Point) weakCopy () Point {
	newpoint := Point{originalX: point.originalX, y: point.y} // creating new points using same 'originalX' backing array.
	newpoint.x = make([]float64, len(point.x)) // making brand new backing array for 'x'.
	for i := range point.x { newpoint.x[i] = point.x[i]} // copying values from 'point's x to 'newpoint's 'x'.
	return newpoint // return reference to that 'newpoint'.
}

/* sets given 'label' (y value) and given 'features' (x parameters) to this point. */
func (p *Point) set (label float64, features[]float64) {
	pointSize := len(features)+1 // increments point length by 1 to accommodate w0 (the bias) later.
	p.x = make([]float, pointSize) // creates a slice of length equal to 'pointSize' to hold the parameters values.
	p.x[0] = 1.0 // sets first parameter is always 1.0.
	for j := 1; j < pointSize; j++ { // for each parameter, starting from second parameter.
		p.x[j] = features[j-1] // set given values.
	}
	p.originalX = p.x // making another reference to the original data in case we make a transformation to x slice.
	p.y = label // set given class.
}

/* any structure that implements these methods can be used by objects of type 'TargetFunction'. therefore, each structure 
should have it's own definition for each method. */
type TargetFunction interface {
	randomInitialize()
	classifyPoint(x []float64) float64 
}

// structure to represent a target linear function f.
type LineFunction struct {
	a, b            float64 // a and b parameter from y = ax + b
	x1, y1, x2, y2  float64 // parameters to represent first and second point used to create linear function f.
}

// initializes f with random x1, y1, x2, y2 values and calculate a and b from a line that passes through them.
func (f *LineFunction) randomInitialize () {
	f.x1, f.y1, f.x2, f.y2 = random(), random(), random(), random()
	f.a = (f.y2 - f.y1) / (f.x2 - f.x1)
	f.b = f.y1 - f.a*f.x1
}

// return 1.0 for points above f or -1.0 for points under or on top of f.
func (f LineFunction) classifyPoint (x []float64) float64 {
	// if point's y value is bigger than f's slope times point's x value plus b, point is above line, else, it's below.
	//    y   >   a*   x   +   b
	if x[2] > f.a*x[1] + f.b { return 1.0 } else { return -1.0 }
}

// structure to hold parameters used in a function that defines a circle.
type circleFunction struct {
	a, b, c float64 // x and y center and radius.
}

func (f *circleFunction) randomInitialize () {
	f.a, f.b, f.c = random(), random(), random()
}

// return 1.0 for points outside f or -1.0 for points inside or on top of f.
func (f circleFunction) classifyPoint (x []float64) float64 {
	if ((x[1]*x[1])-f.a) + ((x[2]*x[2])-f.b) > f.c { return 1.0 } else { return -1.0 }
}

/* creates a new slice with new Points copying point values from given slice of points 'originalPoints' but maintaining 
backing array for attribute 'originalX'. */
func copyPoints (originalPoints []Point) []Point {
	newPoints := make([]Point, len(originalPoints))
	for i := range originalPoints {
		originalPoints[i].weakCopyTo(&newPoints[i])
	}
	return newPoints
}

/* receives a matrix 'xVectors' representing all points and a 'labels' slace representing the classes and returns an slice 
of type Point. In here, w0 is added to the data. */
func createPointsFromData(labels []float64, xVectors [][]float64) []Point {
	points := make([]Point, len(xVectors)) // creating an slice full of points with all value defaulting to zeros.
	for i := range points { // for each index i in points.
		points[i].set(labels[i], xVectors[i]) // set values to this point.
	}
	return points // return slice of points.
}

/* given an 'amount' of points and an amount of parameters 'pointSize' for the points, returns a slice of 'Point's where 
each point is initialized randomly and then classified according to given a given target function 'f'. */
func createPointsAndClassify (amount int, pointSize int, f TargetFunction) []Point {
	points := make([]Point, amount) // creating an slice of points of length equal to 'amount'.
	pointSize += 1 // increments 'pointSize' by 1 to accommodate w0 (the bias) later.
	var p *Point // a pointer to be used to reference each point.
	for i := range points { // for each index i in points.
		p = &points[i] // reference that point.
		p.x = make([]float64, pointSize) // creates a slice of length equal to 'pointSize' to hold the parameters values.
		p.x[0] = 1.0 // sets first parameter is always 1.0.
		for j := 1; j < pointSize; j++ { // for each parameter, starting from second parameter.
			p.x[j] = random() // sets parameters to a random value.
		}
		p.originalX = p.x // making another reference to the original data in case we make a transformation to x slice.
		p.y = f.classifyPoint(p.x) // classify that point according to given 'f'.
	}
	return points // return slice of points.
}

// returns the dot product between a given 'weights' vector and a given point 'p'.
func dotProduct (v1 []float64, v2 []float64) float64 {
	sum := float64(0) // variable to hold the dot product of the weights and the point.
	for j, v := range v1 { // for each parameter in the 'weights' vector.
		sum += v * v2[j] // multiplies the 'weights' vector parameter with the point parameter and add to 'sum'.
	}
	return sum
}

/* given a 'weightVector' and a point 'p', if the dot product result is bigger than zero but the point classification is not 
bigger than zero, in other words, if the class estimated by the weights is not the correct class known to the point, 
return true, if estimated class is correct, returns false. */
func isMissClassFied (weightVector []float64, p Point) bool {
	if (dotProduct(weightVector, p.x) > 0.0) != (p.y > 0.0) { return true } else { return false }
}

/* executes the perceptron learning algorithm on given 'points' and returns two values: a slice of weights, having the same 
length as the amount of parameters found in the given 'points' and returns the amount of iterations. */
func pla (points []Point, initialWeights []float64, maxIterations int) ([]float64, int, float64) {

	var weightVector []float64 /* a slice, which length will be the amount of point parameters, that will hold the weights to 
	best classify the given 'points'. */
	if initialWeights != nil  && len(initialWeights) == len(points[0].x) { // if any given weights with correct length.
		weightVector = initialWeights // keep the given weights.
	} else { // if no given weights to be used as initial weights.
		weightVector = make([]float64, len(points[0].x)) // creates a new set of weights which defaults to zeros.
	}
	var missClassifiedPoints []int // slice to be used to hold the miss classified points indexes at each iteration.
	iterationCounter := 0 // counter for the amount of iterations.
	minError := len(points) // initializing the amount of error in sample as the amount of points in the sample.
	weightsMinError := make([]float64, len(weightVector)) // weights for the iteration with lowest in sample error amount.

	for { // while true.
		missClassifiedPoints = make([]int, 0) // creates an empty slice to be used for holding miss classified points.

		for i := range points { // for each index i in points.
			/* if the dot product result is bigger than zero but the point classification is not bigger than zero. In other 
			words, if the class estimated by the weights is not the correct class known to the point. */
			if isMissClassFied(weightVector, points[i]) {
				// adds the point's index to the slice of miss classified points.
				missClassifiedPoints = append(missClassifiedPoints, i)
			}
		}

		inError := len(missClassifiedPoints) // retrieving amount of miss classified points at current iteration.
		if inError < minError { // if we found a lower in sample error amount.
			minError = inError // copy that in error amount and we also have to copy the 'weightVector' at that time.
			for j, v := range weightVector { // for each weight in 'weightVector'.
				weightsMinError[j] = v // copy weight to the vector saving values at the iteration with lowest error.
			}
			// if the amount of miss classified points is zero or we ran too many iterations, break out of the loop.
			if inError == 0 { break }
		}
		// if iterationCounter % 10000 == 0 {fmt.Println(iterationCounter)}
		if iterationCounter == maxIterations { break } // if we ran too many iterations, break out of the loop.
		iterationCounter++ /* increment iteration counter. increments should happen only if current iteration has more than
		zero miss classified points, which means this iteration will be used to adjust the weights. */

		/* randomly select the index of a miss classified point. That index is related to the whole points slice. Big N. The 
		random index position inside 'missClassifiedPoints' is limited by the length of 'missClassifiedPoints' itself. 
		Intn(arg) returns an integer inside the interval [0, arg[ . */
		random := missClassifiedPoints[rand.Intn(len(missClassifiedPoints))]
		for j := range weightVector { // for each parameter in the 'weightVector'.
			// updates the weight parameter by adding point parameter multiplied by its class (either +1.0 or -1.0). 
			weightVector[j] += points[random].y * points[random].x[j]
		}
		// fmt.Println(iterationCounter, weightVector, missClassifiedPoints)
	}	
	return weightsMinError, iterationCounter, 
		float64(minError)/float64(len(points)) // return the 'weighVector' and the amount of iterations
}

/* returns the disagreement between given target function 'f' and g, in form of weight vector 'weights', given an 2D 
'interval' and a 'step' as distance between sample points across the whole interval. This function samples the whole 2D 
interval with points far apart by given 'step' and then counts how many times f classifies a point differently than g, in 
form of the signal of the dot product between given weight vector and a sample point, after iterating over all sample 
points. */
func disagreementBetweenFandG (f TargetFunction, weights []float64, interval [][]float64, step float64) float64 {

	/* now I try to make the 'interval' and the 'step' stay inside integer domain to avoid adding decimal numbers. Adding 
	decimal usually gets loss of precision. So here I check if step is a number below 1.0, in which case i multiply it by 
	pow(10, power) where power is a number that puts step*pow(10,power) above 1.0. If power happens to become above 0, we
	multiply the interval and the step by pow(10, power). This way we can use only integers in the nested loops that will 
	sample the interval. */
	roundFactor := 1.0 // will be used to avoid calls to math.round().
	power := 0 // used as exponent for '10' to remove decimals from 'step'.
	for float64(step)*math.Pow10(power) < 1.0 { // while we can multiply 'step' by 10**'power' and it remains below 1.0.
		power++ // increment power to try to make 'step' reach or pass 1.0.
	}
	// if step is below 10 and the rest of its division by 10 of its format transformed to integer is zero.
	if step < 10 && int(step*math.Pow10(power)) % 10 == 0 {
		// due to precision loss, a number that was supposed to be above 1.0 happened to be below. so we roll back 'power'
		power--
	}
	// 'roundFactor' can now be used to bring numbers to integer domain to avoid rounding due to loss of precision.
	if (power > 0) {
		roundFactor = math.Pow10(power)
		step *= roundFactor
		for i := range interval { // for each value in the interval.
			for j := range interval[i] {
				// multiply it with the 'roundFactor' so we can use a 'step' that isn't below 1.0.
				interval[i][j] *= roundFactor
			}
		} /* without making this change we would have to use 'float64(math.Round(float64(i+step)*100)/100)' to update i and j 
		and  this is costly to run in each iteration. */
	}

	disagreementCounter := 0 // counter for the amount of sample points f will disagree with g.
	dummyPoint := Point{x: []float64{1.0, 0.0, 0.0}} // point created to hold temporary values.
	// calculating disagreement.
	for i := interval[0][0]; i <= interval[0][1]; i += step { // for each sample point x value .
		for j := interval[1][0]; j <= interval[1][1]; j += step { // for each sample point y value.
			// sets dummy point's x and y value.
			dummyPoint.x[1], dummyPoint.x[2], dummyPoint.y = i/roundFactor, j/roundFactor, f.classifyPoint(dummyPoint.x)
			// if the dot product's signal differ from f classification's signal.
			if isMissClassFied(weights, dummyPoint) {
				disagreementCounter++ // increment disagreement counter.
			}
		}
	}
	// amount of sample points to sample the whole interval.
	sampleSize := (int((interval[0][1] - interval[0][0])/step) + 1) * (int((interval[1][1] - interval[1][0])/step) + 1)
	// fmt.Println(sampleSize, disagreementCounter, step, roundFactor)

	// returns the amount of disagreed points divided by the amount of sample points.
	return float64(disagreementCounter)/float64(sampleSize)
}

/* Runs perceptron learning algorithm experiment repeated an amount of times given by 'amountOfRuns', each experiment using 
given 'amountOfPoints', each point with length as big as given 'pointSize'. Each experiment consist of creating a target 
function, creating points, classify all points using the target function, learn the function using PLA and calculate the 
divergence between the target function and learned function. In the end, it prints the average amount of iterations the pla 
took after all runs and the average divergence. */
func plaRuns (amountOfRuns, amountOfPoints, pointSize int, initializedByLinearGression bool) {
	// printing time when this function finishes.
	start := time.Now()
	defer printExecutionTime(start)

	fmt.Println("-- Perceptron Learning Algorithm runs")
	
	// will hold the sum of iterations and disagreements for all executions and both will be divided by 'amountOfRuns' later.
	averageIterations, averageDisagreements := 0.0, 0.0
	var weights []float // weightVector used to hold the result for each PLA or an initial set of weights used in the PLA.
	var iterations int // used to hold the amount of iterations each PLA took.
	f := &LineFunction{} // creates a linear target function object.

	for i := 0; i < amountOfRuns; i++ { // for each execution.
		f.randomInitialize() // randomly initializes value for target function.
		// fmt.Printf(">> target function f:\na=%f, b=%f\n%f\t%f\n%f\t%f\n---------\n", f.a, f.b, f.x1, f.y1, f.x2, f.y2)

		// creates a slice of randomly initialized and classified point.
		points := createPointsAndClassify(amountOfPoints, pointSize, f)
		// for i, p:= range points {
		// 	// fmt.Println(i, p)
		// 	fmt.Printf("%f\t%f\t%d\t%d\n", p.x[1], p.x[2], i, int(p.y))
		// }

		if initializedByLinearGression == true { // if set to use linear regression to give an initial set of weights.
			weights = linearRegression(points) // run linear regression.
		} else { // if set to initialize PLA with a zeroed vector of weights/
			weights = nil // keep weights uninitialized as the PLA code will set itself a new one started at zeros.
		}
		/* executes the perceptron learning algorithm with an initial set of weights. returns a set of adjusted weights and 
		amount of iterations that the PLA took to finish. */
		weights, iterations, _ = pla(points, weights, 10000000)
		averageIterations += float64(iterations) // accumulate amount of iterations taken for the PLA.
		// fmt.Println(weights, iterations)
		// // w0 + w1x + w2y = 0;  slope = -(w0/w2)/(w0/w1);  intercept = -w0/w2
		// fmt.Printf(">> hypothesis function h:\n0.0\t%f\n%f\t0.0\n---------\n", 
		// 	-weights[0]/weights[2], -weights[0]/weights[1])
		
		// calculates the probability P[f(x) != g(x)]. probability that a random x disagree between f and g.
		disagreemnt := disagreementBetweenFandG(f, weights, [][]float64{{-1.0, 1.0}, {-1.0, 1.0}}, 0.01)
		averageDisagreements += disagreemnt // accumulates the probability P[f(x) != g(x)]
		// fmt.Println("disagreement:", disagreemnt)
	}
	fmt.Printf("average amount of iterations: %f\naverage percentage of disagreements: %f\n", 
		averageIterations/float64(amountOfRuns), averageDisagreements/float64(amountOfRuns)) // print averages.
}

// simple function to print a 2D array line by line and an name for simple visual identification.
func printMatrix (m [][]float64, name string) {
	fmt.Println(name+":------") // wraps matrix with given name.
	for _, v := range m { // for each value in m. if m is a 2D array, each value will be a 1D array.
		fmt.Println(v) // print value.
	}
	fmt.Println(name+":------") // wraps matrix with given name.
}

/* Calculates the linear regression using a set of given 'points' and returns a vector of weights (as a 1D array/slice) 
having the same length as the amount of parameters a point has. */
func linearRegression (points []Point) []float64 {
	// linear regression formula
	// X+ = ((XTX)(-1))*XT; X = 'points.x'.
	// w = (X+)*y; y = 'points.y'
	// w = ((XTX)(-1))*(XT*y); another way to compute w using less memory.

	amountOfParameters := len(points[0].x) // amount of parameter per point.
	var sum float64 // accumulator for inner products used in following matrix multiplications. 

	// initializing square matrices which sizes are 'amountOfParameters' by 'amountOfParameters'.
	XTX := make([][]float64, amountOfParameters) // result for XT*X.
	XTXinverse := make([][]float64, amountOfParameters)
	// choleskyL := make([][]float64, amountOfParameters)
	// choleskyLT := make([][]float64, amountOfParameters)
	// LDLT := make([][]float64, amountOfParameters)
	tempMatrix := make([][]float64, amountOfParameters)
	for i := range XTX {
		XTX[i] = make([]float64, amountOfParameters)
		XTXinverse[i] = make([]float64, amountOfParameters)
		// choleskyL[i] = make([]float64, amountOfParameters)
		// choleskyLT[i] = make([]float64, amountOfParameters)
		// LDLT[i] = make([]float64, amountOfParameters)
		tempMatrix[i] = make([]float64, amountOfParameters)
	}

	// XT*X
	// calculating cell in diagonal and in the first column/row in a result matrix that is symmetric.
	XTX[0][0] = float64(len(points)) // cell A(0,0). inner product between row w0 of XT and colum w0 of X.
	var columSum float64 // XT first row dot product with any column in X is the same a the sum of all values in that column.
	var diagonalSum float64 // row j in XT dot product with column i in X, where i=j, is the same as column i dot column i.
	var value float64 // temporary copy of a single cell value.
	for j := 1; j < amountOfParameters; j++ { // for each row starting at second row in 'XTX'.
		columSum = float64(0) // initialize sum of all values in the same column.
		diagonalSum = float64(0) // initialize sum of all squared values in the same column.
		for i := range points { // for each 'Point' in 'points'.
			value = points[i].x[j] // temporary copy of vale in index j. XT is transpose of 'points' but 'points' in X.
			columSum += value // accumulate column's values
			diagonalSum += value * value // accumulate column's squared values.
		}
		XTX[0][j] = columSum // sum of all values in the same column of X stored in XT's first row.
		XTX[j][0] = columSum // sum of all values in the same column of X stored in XT's first column.
		XTX[j][j] = diagonalSum // sum of all squared values in the same column of X stored in XT's diagonal.
	}
	// calculating cells out of diagonal and out of first column/row. This loops runs through half of a matrix.
	for j := 1; j < amountOfParameters; j++ { // for each row starting by second row in 'XTX'.
		for i := j+1; i < amountOfParameters; i++ { // for each column starting at row j+1 in 'XTX'.
			sum = float64(0) // initialize accumulator for dot product.
			for k := range points { // for each 'Point' in 'points'.
				sum += points[k].x[i] * points[k].x[j] // dot product between two different columns in X.
			} // 'points' is instantiated in a row by row manner so we can't get a reference to any X's columns directly.
			XTX[j][i] = sum // storing in upper half.
			XTX[i][j] = sum // storing in lower half.
		}
	}
	// printMatrix(XTX, "XTX")

	// cholesky decomposition
	// for j := 0; j < amountOfParameters; j++ { // for all rows j.
	// 	for i := 0; i <= j; i++ { // for indexes i that keeps the loop in the lower half including the diagonal.
	// 		if i == j { // for values to be stored in the diagonal.
	// 			sum = float64(0)
	// 			for k := 0; k < i; k++ {
	// 				sum += choleskyL[j][k] * choleskyL[j][k]
	// 			}
	// 			choleskyL[j][j] = float64(math.Sqrt( float64(XTX[j][j] - sum) ))
	// 			choleskyLT[j][j] = choleskyL[j][j]
	// 		} else { // for values to be stored out of the diagonal.
	// 			sum = float64(0)
	// 			for k := 0; k < i; k++ {
	// 				sum += choleskyL[i][k] * choleskyL[j][k]
	// 			}
	// 			choleskyL[j][i] = (XTX[j][i] - sum)/choleskyL[i][i]
	// 			choleskyLT[i][j] = choleskyL[j][i]
	// 		}
	// 	}
	// }
	// // printMatrix(choleskyL, "choleskyL")

	// inversing XT*X.
	// LU decomposition storing values in place. result stored inside XT*X matrix, the 'XTX'.
	pivot := 0.0 // 
	for j := 0; j < amountOfParameters; j++ { // for each row in 'XTX'.
		for i := 0; i < j; i++ { // for each column until row j-1 in 'XTX'.
			pivot = XTX[j][i]/XTX[i][i] // pivot is cell divided by diagonal's value at current column. 
			// LDLT[j][i] = pivot
			// XTX[j][i] = 0
			XTX[j][i] = pivot // store pivot in place.
			for k := i+1; k < amountOfParameters; k++ { // for each non zero value k at row j.
				XTX[j][k] -= pivot*XTX[i][k] // subtract value k by 'pivot' times diagonal's k value. 
			} // this subtracts lines to create zeros in the lower half.
		}
	}
	// printMatrix(LDLT, "LDLT")
	// printMatrix(XTX, "XTX")

	// forward elimination. L*TempM = B; where B is identity matrix. XTX has the lower half L.
	b := 0.0 // did not instantiate an identity matrix b because it's too easy to represent it as an if-else.
	for j := range tempMatrix { // for each row in 'tempMatrix'.
		for i := range tempMatrix[j] { // for each column in 'tempMatrix'.
			if i == j { b = 1.0 } else { b = 0.0 } // represents identity matrix b as an if-else.
			sum = 0.0 // accumulator for dot product.
			for k := 0; k < j; k++ { // for each k until row j-1.
				sum += XTX[j][k]*tempMatrix[k][i] // dot product with all known values in 'tempMatrix'.
			}
			// identity matrix cell value minus dot product with known values divided by diagonal's value in L, which is 1.
			tempMatrix[j][i] = b - sum
		}
	}
	// printMatrix(tempMatrix, "tempMatrix")

	// backward elimination. U*Inv(L) = TempM. result is inv(XT*X). XTX has the upper half U.
	for j := amountOfParameters-1; j > -1; j-- { // for each row in 'XTXinverse' starting at the last row. 
		for i := 0; i < amountOfParameters; i++ { // for each column in 'XTXinverse'. 
			sum = 0.0 // accumulator for dot product.
			for k := j+1; k < amountOfParameters; k++ { // for each k starting at row j+1.
				sum += XTX[j][k]*XTXinverse[k][i] // dot product with all known values in 'XTXinverse'.
			}
			// 'tempMatrix' cell value minus dot product with known values divided by diagonal's value in U.
			XTXinverse[j][i] = (tempMatrix[j][i] - sum)/XTX[j][j]
		}
	}
	// printMatrix(XTXinverse, "XTXinverse")

	// XT*y . this step reduces memory footprint as this result is a vector sized 'amountOfParameters'.
	XTy := make([]float64, amountOfParameters) // instantiate result vector.
	for j := 0; j < amountOfParameters; j++  { // for each row in 'XT'.
		sum = 0.0 // accumulator for dot product.
		for i := range points { // for each column i in XT.
			sum += points[i].x[j] * points[i].y // dot product between column of XT and vector y.
		}
		XTy[j] = sum
	}

	// w = inv(XT*X) * XT*y
	weightVector := make([]float64, amountOfParameters) // result of linear regression.
	for j := range XTXinverse { // for each row j in 'XTXinverse'.
		sum = 0.0 // accumulator for dot product.
		for i, value := range XTXinverse[j] { // for each column i in 'XTXinverse'.
			sum += value * XTy[i] // dot product between 'XTXinverse' and vector 'XTy'.
		}
		weightVector[j] = sum
	}

	return weightVector
}

/* flips signs of a number of randomly selected points inside given slice of 'points' that equals to the amount of points, 
inside the slice, times given 'noiseRate'. But only if that amount of selected points is at least one. */
func addNoise(noiseRate float64, points []Point) {

	if noiseRate > 0.0 { // if rate is bigger than 0.

		amountOfPoints := len(points)
		amountOfPointsToFlip := int( float64(amountOfPoints)*noiseRate) // amount of points to have sign flipped.

		if amountOfPointsToFlip > 0 { // if amount of points to flip signs is bigger than zero.
			// we will randomly generate 'amountOfPointsToFlip' integers without repetition. 
			// We will create a slice with all the possible indexes, shuffle it and get first 'amountOfPointsToFlip' indexes.
			randomPicks := make([]int, len(points)) // instantiate a slice as long as the slice of points.
			for i := range randomPicks { // for each index in 'randomPicks'.
				randomPicks[i] = i // store i inside that index i. this way we have all indexes available to pick.
			}
			// fmt.Println("randomPicks:", randomPicks)

			/* here we shuffle the slice. shuffle is just a bunch of swaps inside the slice. because we will only use the 
			first 'amountOfPointsToFlip' indexes, we will only shuffle until position equal to 'amountOfPointsToFlip'-1. */
			var randomIndex, temp int
			for i := 0; i < amountOfPointsToFlip; i++ { // for each index i until 'amountOfPointsToFlip'-1.
				randomIndex = rand.Intn(amountOfPoints) // generates a random index.
				temp = randomPicks[i] // saves value ate index i inside temp.
				randomPicks[i] = randomPicks[randomIndex] // overwrites value at random index to index i.
				randomPicks[randomIndex] = temp // overwrites value saved in temp to value at random index.
			}
			// fmt.Println("randomPicks:", randomPicks)

			/* flipping signs of the points positioned at the indexes of the first 'amountOfPointsToFlip' shuffled indexes 
			in the 'randomPick' slice. */
			for i := 0; i < amountOfPointsToFlip; i++ { // for each index i until 'amountOfPointsToFlip'-1
				points[i].y = points[i].y*(-1) // flip sign for point at index i.
			}
		}
	}
}

/* returns the percentage of points that a given 'weightVector' disagrees on classifications with f. */
func missClassifiedRate (weightVector []float64, points []Point) float64 {
	miss := 0 // counter for the amount of times the linear regression does not give the same class as target function f.
	for i := range points { // for each point in given 'points' slice/array.
		/* if the dot product result is bigger than zero but the point classification is not bigger than zero. In other 
		words, if the class estimated by the weights is not the correct class known to the point. */
		if isMissClassFied(weightVector, points[i]) {
			miss++ // it's a miss. the linear regression does not give the same class the point has.
		}
	}
	return float64(miss)/float64(len(points)) // returning amount of miss above amount of points.
}

/* Runs linear regression experiment repeated 'amountOfRuns' times, each experiment using given 'amountOfPoints', each point 
with length as big as given 'pointSize'. Each experiment consist of creating a target function, creating points, classify 
all points using the target function, apply linear regression to the points and. */
func linearRegressionRuns (amountOfRuns, amountOfPoints, amountOfPointsOutOfSample, pointSize int, noiseRate float, 
	shouldRandomInitialize bool, target TargetFunction) {
	
	// printing time when this function finishes.
	start := time.Now()
	defer printExecutionTime(start)

	fmt.Println("-- Linear Regression runs")

	// will hold the sum of E_in and E_out for all executions and both will be divided by 'amountOfRuns' later.
	averageE_in, averageE_out := 0.0, 0.0
	var weights []float64 // weightVector used to hold the result for each linear regression.

	var f TargetFunction // reference to target function that will be used in each iteration.
	if target != nil { // if given 'target' exists.
		f = target // use 'target' as f.
	} else { // if it doesn't exist.
		f = &LineFunction{} // creates a linear target function object.
	}

	for i := 0; i < amountOfRuns; i++ { // for each execution.
		if shouldRandomInitialize { // if set to reinitialize f at every iteration.
			f.randomInitialize() // randomly initializes values for target function.
			// fmt.Printf(">> target function f:\na=%f, b=%f\n%f\t%f\n%f\t%f\n---------\n", f.a, f.b, f.x1, f.y1, f.x2, f.y2)
		}

		// creates a slice of randomly initialized and classified point.
		points := createPointsAndClassify(amountOfPoints, pointSize, f)
		// for i, p:= range points {
		// 	// fmt.Printf("%f, ", p.x[1])
		// 	// fmt.Println(i, p)
		// 	fmt.Printf("%f\t%f\t%d\t%d\n", p.x[1], p.x[2], i, int(p.y))
		// }
		addNoise(noiseRate, points) // flips signs of amountOfPoints*noiseRate points.

		weights = linearRegression(points) // executes the linear regression and get the vector of weights resulted from it.
		// fmt.Println("weights", weights)
		// w0 + w1x + w2y = 0;  slope = -(w0/w2)/(w0/w1);  intercept = -w0/w2
		// fmt.Printf(">> hypothesis function h:\n0.0\t%f\n%f\t0.0\n---------\n", -weights[0]/weights[2], -weights[0]/weights[1])
		averageE_in += missClassifiedRate(weights, points) // calculates and accumulates E_in.

		pointsOutOfSample := createPointsAndClassify(amountOfPointsOutOfSample, pointSize, f) // creates a new set of points.
		averageE_out += missClassifiedRate(weights, pointsOutOfSample) // calculates and accumulates E_out.
	}
	fmt.Printf("average E_in: %f\naverage E_out: %f\n", 
		averageE_in/float64(amountOfRuns), averageE_out/float64(amountOfRuns)) // print averages.
}

/* Runs a given 'amountOfRuns' iteration of linear regression, creating a given number 'amountOfPoints' of points, using a 
given 'target' as target function, flipping signs of a 'noiseRate'*'amountOfPoints' number of randomly selected points, 
applying a given 'transformationFunction' with the values of the points and averages the weights of all these executions to 
get an average weight vector to be used in 'amountOfRuns' number of iterations where a given number 
'amountOfPointsOutOfSample' of points is created, a 'noiseRate'*'amountOfPoints' number of randomly selected points out of 
sample have their signs flipped, a 'transformationFunction' is applied to them and an error out of sample is calculated and 
taken an average at the end.*/
func linearRegressionWithTransformationAveragedOut (amountOfRuns, amountOfPoints, amountOfPointsOutOfSample, pointSize int, 
	noiseRate float64, target TargetFunction, transformationFunction func([]float64) []float64) {

	// printing time when this function finishes.
	start := time.Now()
	defer printExecutionTime(start)

	fmt.Println("-- Linear Regression with Transformation")

	var f TargetFunction = target // using given target function.
	var averageWeights, weights []float64 // reference to weights vectors.

	// using points in sample to calculate the linear regression.
	for i := 0; i < amountOfRuns; i++ { // for each execution.
		// creates a slice of randomly initialized and classified point.
		points := createPointsAndClassify(amountOfPoints, pointSize, f)
		addNoise(noiseRate, points) // flips signs of amountOfPoints*noiseRate points.
		for i := range points { // for each point
			points[i].transform(transformationFunction) // apply transformation to point.
		}
		weights = linearRegression(points) // executes the linear regression and get the vector of weights resulted from it.
		// fmt.Println("weights:", weights)
		if len(weights) == len(averageWeights) { // if both have the same length, we can accumulate 'weights'.
			for i, w := range weights { // for each weight in the weight vector
				averageWeights[i] += w // accumulates the weight in the final weight vector.
			}
		} else { // if 'weights' and 'averageWeights' are not the same size, it's because 'averageWeights' is still empty.
			averageWeights = weights // make 'averageWeights' reference 'weights'.
		} // next iteration will make 'weights' reference a new slice anyway.
	}
	// average weights by dividing each accumulated value.
	for i, w := range averageWeights { // for each accumulated weight in the final weight vector.
		averageWeights[i] = w/float64(amountOfRuns) // divide the weight by 'amountOfRuns'.
	}
	fmt.Println("averageWeights:", averageWeights) // print weights to check answer.

	// using points out of sample to calculate E_out.
	averageE_out := 0.0 // will hold the sum E_out for all executions and will be divided by 'amountOfRuns' later.
	for i := 0; i < amountOfRuns; i++ { // for each execution.
		pointsOutOfSample := createPointsAndClassify(amountOfPointsOutOfSample, pointSize, f) // creates a new set of points.
		addNoise(noiseRate, pointsOutOfSample) // flips signs of amountOfPoints*noiseRate points.
		for i := range pointsOutOfSample { // for each out of sample point.
			pointsOutOfSample[i].transform(transformationFunction) // apply transformation to point.
		}
		averageE_out += missClassifiedRate(averageWeights, pointsOutOfSample) // calculates and accumulates E_out.
	}
	fmt.Println("averageE_out:", averageE_out/float64(amountOfRuns)) // prints average.
}

func printExecutionTime (start time.Time) {
	fmt.Println("total time:", time.Since(start))
}

/* prints amount of iterations and last point found by the descent gradient method given a 'function' and it's gradient 
function, a 'learningRate' and a 'threshold' as the difference between values in consecutive iterations as  stopping 
condition and a starting point 'startVector' as slice form. The given 'function' should have one slice as parameters that 
should have the same length as the 'startVector' slice and the 'gradient' function should have both the argument slice 
length and the return slice length the same as the 'startVector' length. */
func gradientDescent (function func([]float64) float64, gradient func([]float64) []float64, 
	startVector []float64, learningRate float64, threshold float64, maxIterations int) {

	// printing time when this function finishes.
	start := time.Now()
	defer printExecutionTime(start)

	fmt.Println("-- Gradient Descent")

	passedMaxIterations := true // flag to be set to false if threshold is hit before iterations reach 'maxIterations'.

	for i := 0; i < maxIterations; i++ { // while amount of iterations is smaller than maximum amount of iterations allowed.
		before := function(startVector) // getting function value with current point.
		walkVector := gradient(startVector) // gets the gradient vector applied to current point.
		// fmt.Println("walkVector:", walkVector)
		
		// updating current point 'startVector'.
		for j, v := range walkVector { // for each value in the gradient vector at current point.
			startVector[j] -= learningRate*v // add that value times 'learningRate' to current point.
		}
		// fmt.Println("startVector:", startVector)
		
		after := math.Abs(function(startVector)) // getting function value with updated point.
		// fmt.Printf("%d. f(x(t)): %v, f(x(t+1)): %v, difference: %v\n", i+1, before, after, after-before)

		if math.Abs(after - before) < threshold { // if difference between function values is smaller than a 'threshold'.
			passedMaxIterations = false // as it stopped before 'maxIterations', set this flag to false.
			fmt.Printf("amount of iterations: %d, final point is %v\n", i+1, startVector) // print results.
			break // break out of the loop.
		}
	}
	if passedMaxIterations { // if we couldn't hit threshold before reaching 'maxIterations'.
		// print results.
		fmt.Printf("after a limit of %d iterations, final point is %v and function value is %v \n", maxIterations, 
			startVector, function(startVector))
	}
}

/* prints amount of iterations and last point found by the descent gradient method given a 'function' and it's gradient 
function, a 'learningRate' and a 'threshold' as the difference between values in consecutive iterations as  stopping 
condition and a starting point 'startVector' as slice form. The given 'function' should have one slice as parameters that 
should have the same length as the 'startVector' slice and the 'gradient' function should have both the argument slice 
length and the return slice length the same as the 'startVector' length. */
func coordinateDescent (function func([]float64) float64, partialDerivatives []func([]float64) float64, 
	startVector []float64, learningRate float64, threshold float64, maxIterations int) {

	// printing time when this function finishes.
	start := time.Now()
	defer printExecutionTime(start)

	fmt.Println("-- Coordinate Descent")

	passedMaxIterations := true // flag to be set to false if threshold is hit before iterations reach 'maxIterations'.

	for i := 0; i < maxIterations; i++ { // while amount of iterations is smaller than maximum amount of iterations allowed.
		before := function(startVector) // getting function value with current point.

		for j, f := range partialDerivatives { // for each coordinate.
			oneCoordinateWalk := f(startVector) // gets the partial derivative value applied to current point.
			// fmt.Println("oneCoordinateWalk:", oneCoordinateWalk)
			
			// updating current point 'startVector'.
			startVector[j] -= learningRate*oneCoordinateWalk // add that value times 'learningRate' to current point.		
			// fmt.Println("startVector:", startVector)
		}

		after := math.Abs(function(startVector)) // getting function value with updated point.
		// fmt.Printf("%d. f(x(t)): %v, f(x(t+1)): %v, difference: %v\n", i+1, before, after, after-before)

		if math.Abs(after - before) < threshold { // if difference between function values is smaller than a 'threshold'.
			passedMaxIterations = false // as it stopped before 'maxIterations', set this flag to false.
			fmt.Printf("amount of iterations: %d, final point is %v\n", i+1, startVector) // print results.
			break // break out of the loop.
		}
	}
	if passedMaxIterations { // if we couldn't hit threshold before reaching 'maxIterations'.
		// print results.
		fmt.Printf("after a limit of %d iterations, final point is %v and function value is %v \n", maxIterations, 
			startVector, function(startVector))
	}
}


/* given a slice of 'points', a 'learningRate', a 'threshold' and a maximum amount of iteration 'maxIterations' to allow 
the logistic regression to run, returns a slice of weights where the logistic regression stopped and its corresponding 
amount of iterations. The logistic regression stops when it converges or when its amount of iterations reaches given 
'maxIterations'. The converge happens when the euclidean distance between consecutive calculated weights is smaller than 
given 'threshold'. 'learningRate' changes the size of the step used to update weights. This function makes a full pass 
through all given points in a random permutation of their order before evaluating the euclidean distance between consecutive 
weight set. */
func logisticRegression (points []Point, learningRate float64, threshold float64, maxIterations int) ([]float64, int) {

	weights := make([]float64, len(points[0].x)) // the initial set of weights, this initialization defaults to all zeros.
	weightsCopy := make([]float64, len(weights)) // another slice to hold a copy of 'weights' before it gets updated.
	passedMaxIterations := true // flag to be set to false if threshold is hit before iterations reach 'maxIterations'.
	iterations := 0 // will be the amount of iterations the logistic regression takes to converge or will be 'maxIterations'.
	amountOfPoints := len(points) // holding integer representing amount of points in the sample.

	for i := 0; i < maxIterations; i++ {
		
		// we will randomly generate a permutation of 1, 2, · · · , N. 
		// First we create a slice with all the possible indexes and shuffle it.
		randomPicks := make([]int, amountOfPoints) // instantiate a slice as long as the slice of points.
		for j := range randomPicks { // for each index in 'randomPicks'.
			randomPicks[j] = j // store i inside that index i. this way we have all indexes available to pick.
		}
		// fmt.Println("randomPicks:", randomPicks)

		// here we shuffle the slice. shuffle is just a bunch of swaps inside the slice
		var randomIndex, temp int
		for j := 0; j < amountOfPoints; j++ { // for each index i.
			randomIndex = rand.Intn(amountOfPoints) // generates a random index.
			temp = randomPicks[j] // saves value ate index i inside temp.
			randomPicks[j] = randomPicks[randomIndex] // overwrites value at random index to index i.
			randomPicks[randomIndex] = temp // overwrites value saved in temp to value at random index.
		}
		// fmt.Println("randomPicks:", randomPicks)

		// making a copy from 'weights'.
		for j, v := range weights {
			weightsCopy[j] = v
		}
		// fmt.Println("weightsCopy", weightsCopy)

		for _, j := range randomPicks { // for each index in the permutation from 1 to len(point).
			point := points[j] // reference point at index j.
			// multiplication for learning rate and gradient on current point.
			a := (learningRate*(-point.y))/(1+math.Exp(point.y*dotProduct(weights, point.x)))
			for k, v := range point.x { // for each parameter in current point.
				weights[k] -= a*v // update 'weights' parameter with the combined multiplication value times x's parameter.
			}
		}
		// fmt.Println("weights    ", weights)

		// calculating the distance between current weights and weights before the update.
		sum := 0.0 // accumulator.
		for j, v := range weights { // for each parameter in the 'weights'.
			sum += math.Pow(weightsCopy[j]-v, 2) // accumulates the squared difference between 'weights' and 'weightsCopy'.
		}
		sum = math.Sqrt(sum) // takes the squared of the accumulated sum of squares.
		// fmt.Println("distance", sum)

		if sum < threshold { // if the distance between current weights and previous weights is smaller than the 'threshold'.
			passedMaxIterations = false // as it stopped before 'maxIterations', set this flag to false.
			iterations = i+1 // set iterations to i+1 as i starts at 0 but 0 is the first iteration.
			break // break out of the loop.
		}
	}

	// because we can't use 'i' here.
	if passedMaxIterations { // if we reached 'maxIterations'.
		iterations = maxIterations // set the amount of iterations to be 'maxIterations'.
	}

	return weights, iterations
}

/* Runs logistic regression experiment repeated an amount of times given by 'amountOfRuns', each experiment using 
given 'amountOfPoints', each point with length as big as given 'pointSize'. Each experiment consist of creating a target 
function, creating a sample of points, classify all points using the target function, apply logistic regression and get its 
set of weights and its amount of iterations, creating another set of points, and calculate the cross entropy error of that 
new sample using the weights gotten from the initial sample. In the end, it prints the average amount of iterations and the 
average cross entropy error for all experiments. */
func logisticRegressionRuns (amountOfRuns, amountOfPoints, amountOfPointsOutOfSample, pointSize int) {
	// printing time when this function finishes.
	start := time.Now()
	defer printExecutionTime(start)

	fmt.Println("-- Logistic Regression runs")

	sumIterations, sumCrossEntropy := 0.0, 0.0
	var weights []float64 // weightVector used to hold the result for each PLA or an initial set of weights used in the PLA.
	var iterations int // used to hold the amount of iterations each logistic regression took.
	f := &LineFunction{} // creates a linear target function object used to create random linear functions.

	for i := 0; i < amountOfRuns; i++ { // for each execution.
		f.randomInitialize() // randomly initializes value for target function.
		// fmt.Printf(">> target function f:\na=%f, b=%f\n%f\t%f\n%f\t%f\n---------\n", f.a, f.b, f.x1, f.y1, f.x2, f.y2)

		// creates a slice of randomly initialized and classified point.
		points := createPointsAndClassify(amountOfPoints, pointSize, f)
		// for i, p:= range points {
		// 	// fmt.Println(i, p)
		// 	fmt.Printf("%f\t%f\t%d\t%d\n", p.x[1], p.x[2], i, int(p.y))
		// }

		// runs logistic regression to get weights where it converged and amount of iterations.
		weights, iterations = logisticRegression(points, 0.01, 0.01, 10000)
		sumIterations += float64(iterations) // accumulate amount of iterations taken for the PLA.
		// fmt.Println(weights, iterations)
		// // w0 + w1x + w2y = 0;  slope = -(w0/w2)/(w0/w1);  intercept = -w0/w2
		// fmt.Printf(">> hypothesis function h:\n0.0\t%f\n%f\t0.0\n---------\n", 
		// 	-weights[0]/weights[2], -weights[0]/weights[1])

		// creates another sample of points using the same linear function 'f'.
		outOfSamplePoints := createPointsAndClassify(amountOfPointsOutOfSample, pointSize, f)

		sum := 0.0 // accumulator.
		for j := range outOfSamplePoints { // for each out of sample points.
			sum += math.Log(1+math.Exp(-outOfSamplePoints[j].y*dotProduct(weights, outOfSamplePoints[i].x))) // accumulating.
		}
		sum /= float64(amountOfPointsOutOfSample) // takes the average of the accumulated error.
		sumCrossEntropy += sum // accumulates cross entropy errors.

	}
	fmt.Printf("average amount of iterations: %v, average cross entropy: %v\n", 
		sumIterations/float64(amountOfRuns), sumCrossEntropy/float64(amountOfRuns)) // print averages.
}

func homeWorks() {
	fmt.Println("Assignments for computer intelligence course ii on UFRJ 2018\n")

	seed := time.Now().Nanosecond() // using the current nanosecond.
	rand.Seed(int64(seed)) // giving a current nanosecond as seed number for the random generator.
	fmt.Println("seed:", seed, "\n")
	// rand.Seed(0)

	pointSize := 2 // amount of parameters used per point.

	///// HW I
	/// 7. and 8.
	plaRuns(1000, 10, pointSize, false)
	/// 9. and 10.
	// plaRuns(1000, 100, pointSize, false)

	///// HW II
	/// 5. and 6.
	linearRegressionRuns(1000, 100, 1000, pointSize, 0.0, true, nil)
	/// 7.
	// plaRuns(100000, 10, pointSize, true)
	/// 8.
	// linearRegressionRuns(1000, 100, 1000, pointSize, 0.1, false, &circleFunction{0.0, 0.0, 0.6} )
	/// 9. and 10.
	// linearRegressionWithTransformationAveragedOut(1000, 1000, 1000, pointSize, 0.1, &circleFunction{0.0, 0.0, 0.6}, 
	// 	func (x []float64) []float64 {
	// 		return []float64{1.0, x[1], x[2], x[1]*x[2], x[1]*x[1], x[2]*x[2]}
	// })

	///// HW III - Tarefa 1
	/// 5. and 6.
	errorSurface := func (params []float64) float64 {
		return math.Pow(params[0]*math.Exp(params[1])-(2*params[1]*math.Exp(-params[0])), 2)
	}
	errorSurfaceGradient := func (params []float64) []float64 {
		return []float64{
			2*(params[0]*math.Exp(params[1])-(2*params[1]*math.Exp(-params[0])))*
				(math.Exp(params[1])+(2*params[1]*math.Exp(-params[0]))),
			2*(params[0]*math.Exp(params[1])-(2*params[1]*math.Exp(-params[0])))*
				(params[0]*math.Exp(params[1])-(2*math.Exp(-params[0])))}
		// return []float64{errorSurfacePartialU(params), errorSurfacePartialV(params)}
	}
	gradientDescent(errorSurface, errorSurfaceGradient, []float64{1.0, 1.0}, 0.1, 1e-14, 1000)
	/// 7.
	// errorSurface := func (params []float64) float64 {
	// 	return math.Pow(params[0]*math.Exp(params[1])-(2*params[1]*math.Exp(-params[0])), 2)
	// }
	errorSurfacePartialU := func (params []float64) float64 {
		return 2*(params[0]*math.Exp(params[1])-(2*params[1]*math.Exp(-params[0])))*
			(math.Exp(params[1])+(2*params[1]*math.Exp(-params[0])))
	}
	errorSurfacePartialV := func (params []float64) float64 {
		return 2*(params[0]*math.Exp(params[1])-(2*params[1]*math.Exp(-params[0])))*
			(params[0]*math.Exp(params[1])-(2*math.Exp(-params[0])))
	}
	partialDerivatives := []func([]float64) float64{errorSurfacePartialU, errorSurfacePartialV}
	coordinateDescent(errorSurface, partialDerivatives, []float64{1.0, 1.0}, 0.1, 1e-14, 15)
	/// 8. and 9.
	logisticRegressionRuns(100, 100, 1000, pointSize)

}