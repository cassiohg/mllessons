// Made by Cássio Holanda Gonçalves. 2018/09/12

package main

import (
	"fmt"
	"time"
	"math"
	"image"
	"image/color"
	"image/png"
	"os"	
	"os/exec"
	"log"
	"strings"
	"strconv"
	"regexp"
	"runtime"
	"sync"
//uses https://www.csie.ntu.edu.tw/~cjlin/libsvm/. Download, make, move svm-train and svm-predict to this file's folder.	
)

// saves a given 'image' encoded as png at given 'filename' path.
func saveImage (image image.Image, filename string) {
	file, err := os.Create(filename)
	if err != nil { log.Fatal(err) }
	if err := png.Encode(file, image); err != nil { file.Close(); log.Fatal(err) }
	if err := file.Close(); err != nil { log.Fatal(err) }
	fmt.Println("image saved at:", filename)
}

/* sets all pixels in given 'image' to given 'color' value.  */
func fillRect (image *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	bounds := image.Bounds()
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			offset := image.PixOffset(i, j)
			image.Pix[offset], image.Pix[offset+1], image.Pix[offset+2], image.Pix[offset+3] = uint8(r), 
				uint8(g), uint8(b), uint8(a)
		}
	}
}

/* returns the weighted average between given value 'a' and given value 'b'. The weight applied is given 'strength' to 
value 'a' and 1-'strength' to value b. */
func addChannelColor (a uint8, b uint8, strength float64) uint8 {
	return uint8(float64(a)*strength + float64(b)*(1-strength))
}

/* sets all pixels in given 'image' to the weighted average between given 'color' and the pixel found in every position of 
that image, maintaining pixel's alpha */
func fillRectAdding (image *image.RGBA, color color.RGBA) {
	r, g, b, _ := color.RGBA()
	bounds := image.Bounds()
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			offset := image.PixOffset(i, j)
			image.Pix[offset  ] = addChannelColor(uint8(r), image.Pix[offset  ], 0.5)
			image.Pix[offset+1] = addChannelColor(uint8(g), image.Pix[offset+1], 0.5)
			image.Pix[offset+2] = addChannelColor(uint8(b), image.Pix[offset+2], 0.5)
		}
	}
}

/* sets pixel in given 'image' at position given 'x' and given 'y' to the weighted average between its color and given 
'color', maintaining pixel's alpha. */
func addColorAt(x, y int, image *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	offset := image.PixOffset(x, y)
	image.Pix[offset  ] = addChannelColor(uint8(r), image.Pix[offset  ], 0.25)
	image.Pix[offset+1] = addChannelColor(uint8(g), image.Pix[offset+1], 0.25)
	image.Pix[offset+2] = addChannelColor(uint8(b), image.Pix[offset+2], 0.25)
	image.Pix[offset+3] = uint8(a)

}


/* fills a square of 9 pixels around point specified by given 'x' and given 'y' with given 'color' in given 'image'. */
func setPointAround(x, y int, image *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	bounds := image.Bounds()
	minX, maxX, minY, maxY, thickness := 0, 0, 0, 0, 1
	if bounds.Min.X < x-thickness { minX = x-thickness } else { minX = bounds.Min.X }
	if bounds.Max.X > x+thickness { maxX = x+thickness } else { maxX = bounds.Max.X }
	if bounds.Min.Y < y-thickness { minY = y-thickness } else { minY = bounds.Min.Y }
	if bounds.Max.Y > y+thickness { maxY = y+thickness } else { maxY = bounds.Max.Y }
	for j := minY; j <= maxY; j++ {
		for i := minX; i <= maxX; i++ {
			offset := image.PixOffset(i, j)
			image.Pix[offset], image.Pix[offset+1], image.Pix[offset+2], image.Pix[offset+3] = uint8(r), 
				uint8(g), uint8(b), uint8(a)
		}
	}
}

/* returns a string representing the whole file content at given 'path'. */
func readFileAsString (filename string) string {
	file, err := os.Open(filename)
	if err != nil { log.Fatal(err) }
	data := make([]byte, 0)
	chunk := make([]byte, 1000)
	for {
		count, err := file.Read(chunk)
		if count == 0 { break }
		if err != nil { log.Fatal(err) }
		data = append(data, chunk[:count]...)
	}
	return string(data)
}

/* converts KEEL data file format given at path 'readFileName' to LIBSVM file format and writes it to given path 
'writeFileName'. */
func convertDataToLibsvmFormat (readFileName, writeFileName string) {
	lines := strings.Split(readFileAsString(readFileName), "\n")
	outStringSlice := ""
	for i := 7; i < len(lines)-1; i++ {
		line := strings.Split(lines[i], ",")
		labelNumber, _ := strconv.ParseFloat(line[2], 64)
		labelString := strconv.FormatFloat(labelNumber, 'g', 1, 64)
		if labelNumber > 0 { labelString = "+"+labelString}
		outStringSlice += labelString+" 1:"+line[0]+" 2:"+line[1]+"\n"
	}
	file, err := os.Create(writeFileName)
	if err != nil { log.Fatal(err) }
	if _, err := file.WriteString(outStringSlice); err != nil { log.Fatal(err) }
	if err := file.Close(); err != nil { log.Fatal(err) }
	fmt.Println("data points file saved at:", writeFileName)
}

/* saves all given 'image' pixels, with x and y values divided by given 'scale' as points using LIBSVM file format to path 
'pathToGridFile' */
func saveGridToFile (pathToGridFile string, image image.Image, scale float64) {
	outStringSlice := make([]string, 0)
	bounds := image.Bounds()
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			outStringSlice = append(outStringSlice, fmt.Sprintf("1 1:%v 2:%v", float64(i)/scale, float64(j)/scale))
		}
	}
	file, err := os.Create(pathToGridFile)
	if err != nil { log.Fatal(err) }
	if _, err := file.WriteString(strings.Join(outStringSlice, "\n")); err != nil { log.Fatal(err) }
	if err := file.Close(); err != nil { log.Fatal(err) }
	fmt.Println("grid points file saved at:", pathToGridFile)	

}

/* returns a slice of labels and a matrix of attributes, where each line is a data point, after reading LIBSVM data file 
at given 'pathToDataFile' and skiping the first 'skipedFirstLines'. */
func readPoints (pathToDataFile string, skipedFirstLines int) ([]float64, [][]float64) {
	lines := strings.Split(readFileAsString(pathToDataFile), "\n")
	lines = lines[skipedFirstLines:len(lines)-1]
	labels := make([]float64, len(lines))
	xs := make([][]float64, len(lines))
	for i := range xs {
		xs[i] = make([]float64, 2)
	}
	for i := range lines {
		line := strings.Split(lines[i], " ")
		n, _ := strconv.ParseFloat(line[0], 64)
		labels[i] = float64(n)
		for j := 1; j < len(line); j++ {
			if line[j] != ""{
				f, _ := strconv.ParseFloat(line[j][strings.LastIndex(line[j], ":")+1:], 64)
				xs[i][j-1] = f
			}
		}

	}
	return labels, xs
}

/* returns an image which boundaries are wide and tall enough to fit all 2-attribute data points in given 'xs' matrix 
but multiplying all point attributes with given 'scale'. This makes the image bigger, the bigger the scale. The image's 
lower boundaries are lower than the points with lowest values. The returned image most certainly won't start at (0,0), it 
will depend on the point's values. */
func buildImageBigEnoughToFitPoints (xs [][]float64, scale float64) *image.RGBA {
	minX, maxX, minY, maxY := xs[0][0], xs[0][0], xs[0][1], xs[0][1]
	for i := range xs {
		if maxX < xs[i][0] { maxX = xs[i][0] } else if minX > xs[i][0] { minX = xs[i][0] }
		if maxY < xs[i][1] { maxY = xs[i][1] } else if minY > xs[i][1] { minY = xs[i][1] }
	}
	extraSpace := 0.05
	widthExtra, heightExtra := (maxX-minX)*extraSpace, (maxY-minY)*extraSpace
	image := image.NewRGBA(image.Rect(int(math.Round((minX-widthExtra)*scale)), int(math.Round((minY-heightExtra)*scale)), 
		int(math.Round((maxX+widthExtra)*scale)), int(math.Round((maxY+heightExtra)*scale))))
	fillRect(image, color.RGBA{0, 0, 0, 255})
	return image	
}

/* draw, in given 'image', which positions are specified by 2-attribute data points 'xs' matrix which values are 
multiplied by given 'scale', that should be the same used when creating this given 'image', and paints these points with 
a color that is function of the values found on given 'labels'. */
func drawPointsInImage(labels []float64, xs [][]float64, image *image.RGBA, scale float64) {
	for i := range xs {
		setPointAround(int(math.Round(xs[i][0]*scale)), int(math.Round(xs[i][1]*scale)), image, 
			color.RGBA{(255/2)*uint8(1+labels[i]), (255/2)*uint8(1-labels[i]), 0, 255})
	}
}

/* reads points from file at given 'pathToDataFile', builds an image with theses points multiplying their values with given 
'scale', saves that image at path 'pathToDistributionImageFile' and returns a pointer to that image. */
func drawPointsFromFile (pathToDataFile string, pathToDistributionImageFile string, scale float64) *image.RGBA {
	labels, xs := readPoints(pathToDataFile, 0)
	image := buildImageBigEnoughToFitPoints(xs, scale)
	drawPointsInImage(labels, xs, image, scale)
	saveImage(image, pathToDistributionImageFile)
	return image
}



/*  Builds an image to show points read from path 'pathToDataFile', painting it's background pixels with a color that is
function of the labels found in file at path 'pathToPredictionFile'. That prediction file should be the one returned by 
LIBSVM svm-predict program when it read the grid points file as points to predict. Saves that image at path 
'pathToModelAreaImageFile' and returns a points to this image. This is the function that draw the svm's 
area/contour/decision boundary. */
func drawModelAreaWithPoints(pathToDataFile, pathToPredictionFile, pathToModelAreaImageFile string, 
scale float64) *image.RGBA {
	labels, xs := readPoints(pathToDataFile, 0)
	image := buildImageBigEnoughToFitPoints(xs, scale)

	lines := strings.Split(readFileAsString(pathToPredictionFile), "\n")
	lines = lines[:len(lines)-1]
	labelsGrid := make([]float64, len(lines))
	for i := range lines {
		line := strings.Split(lines[i], " ")
		n, _ := strconv.ParseFloat(line[0], 64)
		labelsGrid[i] = n
	}
	// _, xsGrid := readPoints(pathToGridFile, 0)

	bounds := image.Bounds()
	k := 0
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			addColorAt(i, j,image, color.RGBA{(255/2)*uint8(1+labelsGrid[k]), (255/2)*uint8(1-labelsGrid[k]), 0, 255})
			k++
		}
	}

	drawPointsInImage(labels, xs, image, scale)

	saveImage(image, pathToModelAreaImageFile)
	return image
}

/* draws, in given 'image', support vectors found on file at 'pathToModelFile' which values are multiplied by given 'scale', 
the same 'scale' used to create the given 'image', and saves that 'image' to given path 'pathToModelAreaImageFile'.  */
func drawSupportVectorsOnTopOfPoints(image *image.RGBA, pathToModelFile, pathToModelAreaImageFile string, scale float64) {
	labels, xs := readPoints(pathToModelFile, 9)
	for i := range xs {
		setPointAround(int(math.Round(xs[i][0]*scale)), int(math.Round(xs[i][1]*scale)), image,  color.RGBA{
			(255/2)*uint8(1+labels[i]), 
			(255/2)*uint8(1-labels[i]), 
			((255/2)*uint8(1+labels[i]))+((255/2)*uint8(1-labels[i])), 
			255})
	}
	saveImage(image, pathToModelAreaImageFile)
}

/* executes, synchronously, a command on UNIX systems that calls LIBSVM's svm-train program at given path 'pathToProgram', 
filling path to data file given by 'pathToDataFile' and parameters given by 'combination' of parameters, using the -v 
parameter to make it run cross validation. When program finished, it parses the output to extract the printed accuracy and 
returns it as float64. */ 
func crossValidateCmd (pathToProgram string, pathToDataFile string, combination Combination) float64 {
	cmd := exec.Command("sh", "-c", pathToProgram+" "+combination.buildParametersString(true)+" "+pathToDataFile)
	out, err := cmd.CombinedOutput()
	if err != nil { log.Fatal(err) }
	f, _ := strconv.ParseFloat(regexp.MustCompile(`\d+.?\d+`).FindStringSubmatch(string(out))[0], 64)
	return f/100.0
}

// structure that holds parameters used to fill command call to LIBSVM's svm-train program.
type Combination struct {
	kernel int; c float64; degree int; gamma float64; coefficient float64; kFold int; accuracy float64
}

// comparison function between two Combination structures. Returns true if at least one attribute is different.
func (c1 Combination) isDifferent (c2 Combination) bool {
	if c1.kernel != c2.kernel || c1.degree != c2.degree || c1.kFold != c2.kFold || 
	c1.c != c2.c || c1.gamma != c2.gamma || c1.coefficient != c2.coefficient {
		return true
	} else {
		return false
	}
}

/* returns a string containing the relevant parameters related to combinations kernel type. If given 'vparam' is true, 
it will include the amount of folds. Without the amount of folds in the parameters string, LIBSVM's svm-train will train 
will all points and will not do cross validation. */
func (c Combination) buildParametersString (vparam bool) string {
	str := fmt.Sprintf("-q -m 400 -s 0 -t %v -c %v", c.kernel, c.c)
	switch c.kernel {
		case RBF:
			str += fmt.Sprintf(" -g %v", c.gamma)
		case LINEAR:
			str += ""
		case SIGMOID:
			str += fmt.Sprintf(" -g %v -r %v", c.gamma, c.coefficient)
		case POLY:
			str += fmt.Sprintf(" -d %v -g %v -r %v", c.degree, c.gamma, c.coefficient)
	}
	if (vparam) {
		str += fmt.Sprintf(" -v %v", c.kFold)
	}
	return str
}

/* prints a human friendly string showing all parameters this combination of parameters has, including the accuracy result.*/
func (c Combination) printableParameters () string {
	str := ""
	switch c.kernel {
		case RBF:
			str += fmt.Sprintf("kernel: %v, c: %v, gamma: %v, k-fold: %v, accuracy = ",c.getKernelName(),c.c,c.gamma,c.kFold)
		case LINEAR:
			str += fmt.Sprintf("kernel: %v, c: %v, k-fold: %v, accuracy = ", c.getKernelName(), c.c, c.kFold)
		case SIGMOID:
			str += fmt.Sprintf("kernel: %v, c: %v, gamma: %v, coefficient: %v, k-fold: %v, accuracy = ", 
				c.getKernelName(), c.c, c.gamma, c.coefficient, c.kFold)
		case POLY:
			str += fmt.Sprintf("kernel: %v, c: %v, degree: %v, gamma: %v, coefficient: %v, k-fold: %v, accuracy = ", 
				c.getKernelName(), c.c, c.degree, c.gamma, c.coefficient, c.kFold)
	}
	str += fmt.Sprint(c.accuracy)
	return str
}

/* returns the name of the kernel defined by it's type number. */
func (c Combination) getKernelName () string {
	switch c.kernel {
		case RBF:
			return "RBF"
		case LINEAR:
			return "linear"
		case SIGMOID:
			return "sigmoid"
		case POLY:
			return "polinomial"
		default:
			return "nil"
	}
}

/* defines a number representing each kernels. this coincides with LIBSVM's kernel types numbers. */
const (
	LINEAR 	= 0
	POLY 	= 1 
	RBF 	= 2
	SIGMOID = 3
)

/* builds all combination of parameters to feed LIBSVM's svm-train program found at given path 'pathToTrainProgram', passing 
data file found at given path 'pathToDataFile', and executes the program call is parallel using a number of 
'numberOfThreadsToUse' threads. In the end, the best combination of parameters of each kernel type is returned. */
func tryAllSVMsAndFindBestModel (pathToTrainProgram, pathToDataFile string, numberOfThreadsToUse int) ([]Combination){
	// all values used to generate all possible combinations of these parameters.
	kernels := []int{RBF, POLY, LINEAR, SIGMOID}
	cs := []float64{1.0}
	degrees := []int{3, 4, 5}
	gammas := []float64{0.01, 0.5, 1.0, 10.0}
	coefficients := []float64{0.0, 1.0, 2.0}
	folds := []int{2, 5, 10}

	combinations := make([][]Combination, 4) // LIBSVM has 4 kernel types starting a 0, fortunately. so we don't need a map.
	for i := range combinations { combinations[i] = make([]Combination, 0) }

	// control to amount of parallel threads.
	semaphore := make(chan struct{}, numberOfThreadsToUse)
	// object that will count amount of parallel threads running. it will block execution until all threads finish.
	var wg sync.WaitGroup

	for _, kernel := range kernels {
		for _, c := range cs {
			
			alreadyPassedDegree := false
			for _, degree := range degrees {
				// this IF statement checks whether current kernel can skip this parameter but allows it to pass only once.
				if alreadyPassedDegree == true && (kernel == RBF || kernel == LINEAR || kernel == SIGMOID) { break 
				} else { alreadyPassedDegree = true }
				
				alreadyPassedGamma := false
				for _, gamma := range gammas {
					if alreadyPassedGamma == true && (kernel == LINEAR) { break 
					} else { alreadyPassedGamma = true }
					
					alreadyPassedCoef := false
					for _, coefficient := range coefficients {
						if alreadyPassedCoef == true && (kernel == RBF || kernel == LINEAR) { break 
						} else { alreadyPassedCoef = true }
						
						for _, kFold := range folds {
							// creating an object to hold all parameters.
							combination := Combination{kernel: kernel, c: c, degree: degree, 
								gamma: gamma, coefficient: coefficient, kFold: kFold}
							// increment thread counter.
							wg.Add(1)
							// add dummy thing to buffered channel but this line blocks if channel is full, until it's not.
							semaphore <- struct{}{}
							// starts a new thread adding a copy of 'combination' to its stack.
							go func(combination Combination) {
								defer wg.Done() // decrement thread counter
								defer func(){ <-semaphore }() // remove dummy thing from buffered channel.

								start := time.Now()
								combination.accuracy = crossValidateCmd(pathToTrainProgram, pathToDataFile, combination)
								fmt.Printf("Cross Validation. %v, time: %v\n", combination.printableParameters(), 
									time.Since(start))
								// saving combination object.
								combinations[combination.kernel] = append(combinations[combination.kernel], combination)

							}(combination) // calling function passing 'combination' to it's stack.
						}
					}
				}
			}
		}
	}
	wg.Wait() // this lines blocks execution waiting for counter to reach zero.

	// selecting best parameters combination of each kernel type.
	bestOfEach := make([]Combination, 0)
	for i := range combinations {
		if len(combinations[i]) > 0 {
			bestCombination := combinations[i][0]
			for _, comb := range combinations[i] {
				if comb.accuracy > bestCombination.accuracy{
					bestCombination = comb
				}
			}
			bestOfEach = append(bestOfEach, bestCombination)
			fmt.Println("best accuracy of its kernel type >>>", bestCombination.printableParameters())
		}
	}

	return bestOfEach
}

/* executes, synchronously, a command on UNIX systems that calls LIBSVM's svm-train program at given path 'pathToProgram', 
filling path to data file given by 'pathToDataFile' and parameters given by 'combination' of parameters, not using the -v 
parameter so it can train with the whole data, saving model file at given path 'pathToModelFile'. */ 
func train (pathToTrainProgram, pathToDataFile, pathToModelFile string, combination Combination) {
	fmt.Println("training", combination.printableParameters())
	cmd := exec.Command("sh", "-c", 
		pathToTrainProgram+" "+combination.buildParametersString(false)+" "+pathToDataFile+" "+pathToModelFile)
	_, err := cmd.CombinedOutput()
	if err != nil { log.Fatal(err) }
}

/* executes, synchronously, a command on UNIX systems that calls LIBSVM's svm-predict program at given path 'pathToProgram', 
filling path to test data file given by 'pathToGridFile' (file that should have been created previously by another function 
made by this code, where all image's pixels positions are saved to file using LIBSVM file format), filling path to model 
file given by 'pathToModelFile' and filing output file given by 'pathToPredictionFile'. */
func predictGridPoints (pathToPredictProgram, pathToGridFile, pathToModelFile, pathToPredictionFile string) {
	fmt.Println("predicting to file", pathToPredictionFile)
	cmd := exec.Command("sh", "-c", 
		pathToPredictProgram+" "+pathToGridFile+" "+pathToModelFile+" "+pathToPredictionFile)
	_, err := cmd.CombinedOutput()
	if err != nil { log.Fatal(err) }
}

/* returns a function that transforms a vector like [1.0, x1, x2] to another vector where a transformation of order given 
by 'n' has been applied to it. */
func transformationOfOrderN (n int, x []float64) []float64 {
	x2 := []float64{}
	for i := 0; i <= n; i++ {
		for j := 0; j <= i; j++ {
			x2 = append(x2, math.Pow(x[1], float64(j))*math.Pow(x[2], float64(i-j)))
		}
	}
	return x2
}

/* receives given slice of Points 'points', and returns a slice of slices of Points where the first level represents a 
class and the second level there is a slice containing all points that belong to that class. If given 
'transformationFunction' isn't nil, it applies that transformation to all points. Divings the data this way we would never 
put all data from a same class in the same fold, after diving the data set in k amount of folds, which would be horrible 
for training when that fold becomes be the one left. */
func separatePointsPerClass (points []Point, transformationFunction func ([]float64) []float64) [][]Point {
	pointsDividedByClass := [][]Point{[]Point{}, []Point{}}
	for i := range points {
		if transformationFunction != nil {
			points[i].transform(transformationFunction)
		}
		whichClass := 0 // index for class -1 or +1.
		if points[i].y > 0.0 { whichClass = 1 } // as there are only two classes, class index 0 is -1 and class 1 is +1.
		pointsDividedByClass[whichClass] = append(pointsDividedByClass[whichClass], points[i])
	}
	return pointsDividedByClass
}

/* receives a slice of slices of Points where the first level represents a class and the second level there is a slice 
containing all points that belong to that class and reorganizes the that in another slice of slices of points where the 
first level has length equal to given 'amountOfFolds' and second level has all the points that belong to that fold. A fold
is just a group of equal amount of points belonging to the whole data set. it receives a slice of slices of Points belonging 
to a same class because when we create folds from points coming organized this way, we would never put all data from a same 
class in the same fold, after diving the data set in k amount of folds, which would be horrible for training when that fold 
becomes be the one left. We also take one points every 'amountOfFolds' points, and not 'amountOfFolds' consecutive points, 
in order to avoid making a fold full of points from the same cluster or points. This way we try to pick points as spread 
as possible around the space. */
func createFolds (pointsDividedByClass [][]Point, amountOfFolds int) [][]Point {
	folds := [][]Point{}
	for k := 0; k < amountOfFolds; k++ {
		folds = append(folds, []Point{})
		for i := range pointsDividedByClass {
			for j := 0; j < len(pointsDividedByClass[i]); j += amountOfFolds {
				folds[k] = append(folds[k], pointsDividedByClass[i][j])
			}
		}
		// fmt.Printf("fold-%v, length=%v\n", k, len(folds[k]))
	}
	return folds
}

/* given a set of 'folds' and a function that runs a learning algorithm with one set given of Points (training), evaluates 
error with another set of given Points (validation) and returns the average error found for the second set of Points, given 
by 'trainAndEvaluationFunction', a battery of training and validation is done using one fold as validation and the other 
folds as training data set, which means, cross validation is performed, and the average error for the folds is returned. */
func crossValidate (folds [][]Point, trainAndEvaluationFunction func ([]Point, []Point) float64) float64 {
	averageCrossValidationError := 0.0
	for k := range folds {
		currentFold := folds[k]
		remainingFoldsJoined := []Point{}
		for i := range folds {
			if i != k {
				remainingFoldsJoined = append(remainingFoldsJoined, folds[i]...)
			}
		}
		crossValidationError := trainAndEvaluationFunction(remainingFoldsJoined, currentFold)
		averageCrossValidationError += crossValidationError
		// fmt.Printf("Cross Validation fold %v, error: %v\n", k, crossValidationError)
	}
	averageCrossValidationError /= float64(len(folds))
	return averageCrossValidationError // "returned" value.
}

/* runs PLA using given set 'trainingSet' of Points with no initial set of weight and fixed maximum amount of iterations, 
uses the PLA's final weights and counts amount of miss classified points from given set 'validationSet' of Points and 
returns the average amount of miss classified points. */
func trainPerceptronAndEvaluatesValidationSet (trainingSet []Point, validationSet []Point) float64 {
	weights, _, _ := pla(trainingSet, nil, 20000)
	amountOfMissClassified := 0
	for i := range validationSet {
		if isMissClassFied(weights, validationSet[i]) {
			amountOfMissClassified++
		}
	}
	return float64(amountOfMissClassified)/float64(len(validationSet))	
}

/* given the order 'n' of the transformation applied to given 'points', a copy of the points is created, the points are 
divided by class and then divided in k 'amountOfFolds' folds, a cross validation is called giving a perceptron training and 
evaluation function, the average error returned from the cross validation is set to the address given as 'cvError' and 
in the end the callback is called. This function was made to run in parallel so it doesn't write or change values inside 
the given 'points' and it write it's final value in one of the argument instead of returning it. */
func crossValdidatePerceptronWithTransformationOfOrder(n int, points []Point, amountOfFolds int, 
cvError *float64, callback func ()) {
	defer callback()
	
	fmt.Println("started working on cross validation for transformation of order", n)
	newPoints := copyPoints(points)

	transformationFunction := func (x []float64) []float64 { return transformationOfOrderN(n, x) }
	pointsDividedByClass := separatePointsPerClass(newPoints, transformationFunction)
	
	folds := createFolds(pointsDividedByClass, amountOfFolds)
	
	e := crossValidate(folds, trainPerceptronAndEvaluatesValidationSet)

	fmt.Printf("transformation order: %v, averageCrossValidationError: %v\n", n, e)
	*cvError = e	
}

/* runs cross validation, in parallel using 'numberOfThreadsToUse' amount of threads, using perceptron on transformations 
applied to points read from file at given path 'pathToDataFile', where the first transformation is of order 2 and the last 
is of order given by 'lastTransformationOrderToTry', selects the transformation that gave the highest accuracy and creates 
and saves an image with the area/contour/decision boundary of that transformation in the background of the points read from
given file path. */
func tryPerceptronWithTransformationAndDrawResults (lastTransformationOrderToTry int, pathToDataFile, 
pathToModelAreaImageFile string, scale float64, numberOfThreadsToUse int) {

	amountOfFoldsInCrossValidation := 10
	firstTranformationOrderToTry := 2
	fmt.Println("Trying perceptron after applying transformation to data points.")
	fmt.Printf("We will try transformations from order %v until order %v\n", firstTranformationOrderToTry, 
		lastTransformationOrderToTry)

	labels, xs := readPoints(pathToDataFile, 0)
	points := createPointsFromData(labels, xs)

	semaphore := make(chan struct{}, numberOfThreadsToUse)
	var wg sync.WaitGroup

	// cross validate perceptrons after applying transformation to data points.
	perceptronErrors := make([]float64, lastTransformationOrderToTry+1)
	for i := range perceptronErrors { perceptronErrors[i] = 1.0 } // setting max error for all n-th order transformation.
	for n := firstTranformationOrderToTry; n <= lastTransformationOrderToTry; n++ {
		wg.Add(1); semaphore <- struct{}{}
		go crossValdidatePerceptronWithTransformationOfOrder(n, points, amountOfFoldsInCrossValidation, 
			&perceptronErrors[n], func () { wg.Done(); <-semaphore })
	}
	wg.Wait()

	// selecting lowest error.
	minError, bestOrder := 1.0, 0
	for i := range perceptronErrors {
		if minError > perceptronErrors[i] { minError = perceptronErrors[i]; bestOrder = i }
	}

	// whole data training.
	transformationFunction := func (x []float64) []float64 { return transformationOfOrderN(bestOrder, x) }
	for i := range points {points[i].transform(transformationFunction)}
	weights, _, inError := pla(points, nil, 50000)
	fmt.Printf("Best perceptron. transformation of order=%v, accuracy=%v, in sample Error=%v\n", 
		bestOrder, 1.0-minError, inError)

	//drawing area/contour/decision boundary image.
	image := buildImageBigEnoughToFitPoints(xs, scale)
	bounds := image.Bounds()
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			gridPointParameters := transformationOfOrderN(bestOrder, []float64{1.0, float64(i)/scale, float64(j)/scale})
			label := 0
			if dotProduct(weights, gridPointParameters) > 0 { label = 1 } else { label = -1 }
			addColorAt(i, j,image, color.RGBA{(255/2)*uint8(1+label), (255/2)*uint8(1-label), 0, 255})
		}
	}

	// drawing original points
	drawPointsInImage(labels, xs, image, scale)

	// saving image.
	pathToModelAreaImageFile2 := pathToModelAreaImageFile[:len(pathToModelAreaImageFile)-4]+
		"Order"+fmt.Sprintf("%02d", bestOrder)+pathToModelAreaImageFile[len(pathToModelAreaImageFile)-4:]
	saveImage(image, pathToModelAreaImageFile2)
}

func main() {
	numberOfThreadsToUse := 4
	runtime.GOMAXPROCS(numberOfThreadsToUse)
	fmt.Println("Assignments for computer intelligence course ii on UFRJ 2018\n")

	// path := "/Users/cassiohg/Coding/C++/libsvm-3.23/"
	path := "./"
	originalFormatDataFileName := "banana.dat"
	pathToOriginalFormatFile := path+originalFormatDataFileName
	pathToTrainProgram := path+"svm-train"
	pathToPredictProgram := path+"svm-predict"
	dataFileName := "banana.txt"
	pathToDataFile := path+dataFileName
	pathToDistributionImageFile := "banana.png"
	gridFileName := "bananaGrid.txt"
	pathToGridFile := path+gridFileName
	modelFileName := "bananaModel.txt"
	pathToModelFile := path+modelFileName
	predictionFileName := "bananaPrediction.txt"
	pathToPredictionFile := path+predictionFileName
	pathToModelAreaImageFile := "bananaModelArea.png"

	convertDataToLibsvmFormat(pathToOriginalFormatFile, pathToDataFile)

	scale := 100.0
	//// item 1.
	image := drawPointsFromFile(pathToDataFile, pathToDistributionImageFile, scale) // drawing distribution.
	saveGridToFile(pathToGridFile, image, scale) // saving all image pixels as points to be predicted later by each model.

	//// item 2.
	selectedCombinations := tryAllSVMsAndFindBestModel(pathToTrainProgram, pathToDataFile, numberOfThreadsToUse)

	//// item 3.
	semaphore := make(chan struct{}, numberOfThreadsToUse)
	var wg sync.WaitGroup

	for _, combination := range selectedCombinations {

		// changing path names to add kernel name and make different output files names.
		pathToModelFile2 := pathToModelFile[:len(pathToModelFile)-4]+
			combination.getKernelName()+pathToModelFile[len(pathToModelFile)-4:]
		pathToPredictionFile2 := pathToPredictionFile[:len(pathToPredictionFile)-4]+
			combination.getKernelName()+pathToPredictionFile[len(pathToPredictionFile)-4:]
		pathToModelAreaImageFile2 := pathToModelAreaImageFile[:len(pathToModelAreaImageFile)-4]+
			combination.getKernelName()+pathToModelAreaImageFile[len(pathToModelAreaImageFile)-4:]

		wg.Add(1); semaphore <- struct{}{} // blocks execution until one thread has finished.
		// draw best models in parallel in another thread.
		go func (pathToModelFile, pathToPredictionFile, pathToModelAreaImageFile string, combination Combination) {
			defer func () { wg.Done(); <-semaphore })
			// traing with all points.
			train(pathToTrainProgram, pathToDataFile, pathToModelFile, combination)
			// predict all image pixels to make area/contour/decision boundary image. 
			predictGridPoints(pathToPredictProgram, pathToGridFile, pathToModelFile, pathToPredictionFile)
			// drawing area/contour/decision boundary image.
			image2 := drawModelAreaWithPoints(pathToDataFile, pathToPredictionFile, pathToModelAreaImageFile, scale)
			// drawing support vector.
			drawSupportVectorsOnTopOfPoints(image2, pathToModelFile2, pathToModelAreaImageFile, scale)
		}(pathToModelFile2, pathToPredictionFile2, pathToModelAreaImageFile2, combination)
	}
	wg.Wait() // blocks execution until all threads have finished.

	//// item 4.
	lastTransformationOrderToTryForPeceptron := 10
	tryPerceptronWithTransformationAndDrawResults(lastTransformationOrderToTryForPeceptron, pathToDataFile, 
		pathToModelAreaImageFile, scale, numberOfThreadsToUse)

}
