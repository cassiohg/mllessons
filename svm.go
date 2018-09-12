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
)

func saveImage (img image.Image, filename string) {
	file, err := os.Create(filename)
	if err != nil { log.Fatal(err) }
	if err := png.Encode(file, img); err != nil { file.Close(); log.Fatal(err) }
	if err := file.Close(); err != nil { log.Fatal(err) }
	fmt.Println("image saved at:", filename)
}

func fillRect (img *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	bounds := img.Bounds()
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			offset := img.PixOffset(i, j)
			img.Pix[offset], img.Pix[offset+1], img.Pix[offset+2], img.Pix[offset+3] = uint8(r), uint8(g), uint8(b), uint8(a)
		}
	}
}

func addChannelColor (a uint8, b uint8, strength float32) uint8 {
	return uint8(float32(a)*strength + float32(b)*(1-strength))
}

func fillRectAdding (img *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	bounds := img.Bounds()
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			offset := img.PixOffset(i, j)
			img.Pix[offset  ] = addChannelColor(uint8(r), img.Pix[offset  ], 0.5)
			img.Pix[offset+1] = addChannelColor(uint8(g), img.Pix[offset+1], 0.5)
			img.Pix[offset+2] = addChannelColor(uint8(b), img.Pix[offset+2], 0.5)
			img.Pix[offset+3] = uint8(a)
		}
	}
}

func addColorAt(x, y int, img *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	offset := img.PixOffset(x, y)
	img.Pix[offset  ] = addChannelColor(uint8(r), img.Pix[offset  ], 0.25)
	img.Pix[offset+1] = addChannelColor(uint8(g), img.Pix[offset+1], 0.25)
	img.Pix[offset+2] = addChannelColor(uint8(b), img.Pix[offset+2], 0.25)
	img.Pix[offset+3] = uint8(a)

}

func setPointAround(x, y int, img *image.RGBA, color color.RGBA) {
	r, g, b, a := color.RGBA()
	bounds := img.Bounds()
	xmin, xmax, ymin, ymax, thickness := 0, 0, 0, 0, 1
	if bounds.Min.X < x-thickness { xmin = x-thickness } else { xmin = bounds.Min.X }
	if bounds.Max.X > x+thickness { xmax = x+thickness } else { xmax = bounds.Max.X }
	if bounds.Min.Y < y-thickness { ymin = y-thickness } else { ymin = bounds.Min.Y }
	if bounds.Max.Y > y+thickness { ymax = y+thickness } else { ymax = bounds.Max.Y }
	for j := ymin; j <= ymax; j++ {
		for i := xmin; i <= xmax; i++ {
			offset := img.PixOffset(i, j)
			img.Pix[offset], img.Pix[offset+1], img.Pix[offset+2], img.Pix[offset+3] = uint8(r), uint8(g), uint8(b), uint8(a)
		}
	}
}

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

func readPoints (pathToDataFile string, skipFirstlines int) ([]float64, [][]float64) {
	lines := strings.Split(readFileAsString(pathToDataFile), "\n")
	lines = lines[skipFirstlines:len(lines)-1]
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

func buildImageBigEnoughToFitPoints (xs [][]float64, scale float64) *image.RGBA {
	xmin, xmax, ymin, ymax := xs[0][0], xs[0][0], xs[0][1], xs[0][1]
	for i := range xs {
		if xmax < xs[i][0] { xmax = xs[i][0] } else if xmin > xs[i][0] { xmin = xs[i][0] }
		if ymax < xs[i][1] { ymax = xs[i][1] } else if ymin > xs[i][1] { ymin = xs[i][1] }
	}
	extraSpace := 0.05
	widthExtra, heightExtra := (xmax-xmin)*extraSpace, (ymax-ymin)*extraSpace
	img := image.NewRGBA(image.Rect(int(math.Round((xmin-widthExtra)*scale)), int(math.Round((ymin-heightExtra)*scale)), 
		int(math.Round((xmax+widthExtra)*scale)), int(math.Round((ymax+heightExtra)*scale))))
	fillRect(img, color.RGBA{0, 0, 0, 255})
	return img	
}

func drawPointsInImage(labels []float64, xs [][]float64, img *image.RGBA, scale float64) {
	for i := range xs {
		setPointAround(int(math.Round(xs[i][0]*scale)), int(math.Round(xs[i][1]*scale)), img, 
			color.RGBA{(255/2)*uint8(1+labels[i]), (255/2)*uint8(1-labels[i]), 0, 255})
	}
}

func drawPoints (pathToDataFile string, pathToDistributionImageFile string, scale float64) *image.RGBA {
	labels, xs := readPoints(pathToDataFile, 0)
	img := buildImageBigEnoughToFitPoints(xs, scale)
	drawPointsInImage(labels, xs, img, scale)
	saveImage(img, pathToDistributionImageFile)
	return img
}

func drawModelAreaWithPoints(pathToDataFile, pathToPreditionFile, pathToModelAreaImageFile string, 
scale float64) *image.RGBA {
	labels, xs := readPoints(pathToDataFile, 0)
	img := buildImageBigEnoughToFitPoints(xs, scale)

	lines := strings.Split(readFileAsString(pathToPreditionFile), "\n")
	lines = lines[:len(lines)-1]
	labelsGrid := make([]float64, len(lines))
	for i := range lines {
		line := strings.Split(lines[i], " ")
		n, _ := strconv.ParseFloat(line[0], 64)
		labelsGrid[i] = n
	}
	// _, xsGrid := readPoints(pathToGridFile, 0)

	bounds := img.Bounds()
	k := 0
	for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
		for i := bounds.Min.X; i < bounds.Max.X; i++ {
			addColorAt(i, j,img, color.RGBA{(255/2)*uint8(1+labelsGrid[k]), (255/2)*uint8(1-labelsGrid[k]), 0, 255})
			k++
		}
	}

	drawPointsInImage(labels, xs, img, scale)

	saveImage(img, pathToModelAreaImageFile)
	return img
}

func drawSupportVectorsOnTopOfPoints(img *image.RGBA, pathToModelFile, pathToModelAreaImageFile string, scale float64) {
	labels, xs := readPoints(pathToModelFile, 9)
	for i := range xs {
		setPointAround(int(math.Round(xs[i][0]*scale)), int(math.Round(xs[i][1]*scale)), img,  color.RGBA{
			(255/2)*uint8(1+labels[i]), 
			(255/2)*uint8(1-labels[i]), 
			((255/2)*uint8(1+labels[i]))+((255/2)*uint8(1-labels[i])), 
			255})
	}
	saveImage(img, pathToModelAreaImageFile)
}

func saveGridToFile (pathToGridFile string, img image.Image, scale float64) {
	outStringSlice := make([]string, 0)
	bounds := img.Bounds()
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

func crossValidate (pathToProgram string, pathToDataFile string, combination Combination) float64 {
	cmd := exec.Command("sh", "-c", pathToProgram+" "+combination.buildParametersString(true)+" "+pathToDataFile)
	out, err := cmd.CombinedOutput()
	if err != nil { log.Fatal(err) }
	f, _ := strconv.ParseFloat(regexp.MustCompile(`\d+.?\d+`).FindStringSubmatch(string(out))[0], 64)
	return f/100.0
}

type Combination struct {
	kernel int; c float64; degree int; gamma float64; coefficient float64; kFold int; accuracy float64
}

func (c1 Combination) isDifferent (c2 Combination) bool {
	if c1.kernel != c2.kernel || c1.degree != c2.degree || c1.kFold != c2.kFold || 
	c1.c != c2.c || c1.gamma != c2.gamma || c1.coefficient != c2.coefficient {
		return true
	} else {
		return false
	}
}

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

func (c Combination) printParameters () string {
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

const (
	LINEAR 	= 0
	POLY 	= 1 
	RBF 	= 2
	SIGMOID = 3
)

func findBestModel (pathToTrainProgram, pathToDataFile string, numberOfThreadsToUse int) ([]Combination){
	kernels := []int{RBF, POLY, LINEAR, SIGMOID}
	cs := []float64{1.0}
	degrees := []int{3, 4, 5}
	gammas := []float64{0.01, 0.5, 1.0, 10.0}
	coefficients := []float64{0.0, 1.0, 2.0}
	folds := []int{2, 5, 10}

	// kernels := []int{RBF, LINEAR, SIGMOID}
	// cs := []float64{1.0}
	// degrees := []int{3, 4}
	// gammas := []float64{0.5, 10.0}
	// coefficients := []float64{0.0, 2.0}
	// folds := []int{2}

	combinations := make([][]Combination, 4)
	for i := range combinations { combinations[i] = make([]Combination, 0) }

	semaphore := make(chan struct{}, numberOfThreadsToUse)
	var wg sync.WaitGroup

	for _, kernel := range kernels {
		for _, c := range cs {
			
			alreadyPassedDegree := false
			for _, degree := range degrees {
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
							combination := Combination{kernel: kernel, c: c, degree: degree, 
								gamma: gamma, coefficient: coefficient, kFold: kFold}
							wg.Add(1)
							semaphore <- struct{}{}
							go func(combination Combination) {
								defer wg.Done()
								defer func(){ <-semaphore }()

								start := time.Now()
								combination.accuracy = crossValidate(pathToTrainProgram, pathToDataFile, combination)
								fmt.Printf("Cross Validation. %v, time: %v\n", combination.printParameters(), 
									time.Since(start))
								combinations[combination.kernel] = append(combinations[combination.kernel], combination)
							}(combination)
						}
					}
				}
			}
		}
	}
	wg.Wait()

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
			fmt.Println("best accuracy of its kernel type >>>", bestCombination.printParameters())
		}
	}

	return bestOfEach
}

func train (pathToTrainProgram, pathToDataFile, pathToModelFile string, combination Combination) {
	fmt.Println("training", combination.printParameters())
	cmd := exec.Command("sh", "-c", 
		pathToTrainProgram+" "+combination.buildParametersString(false)+" "+pathToDataFile+" "+pathToModelFile)
	_, err := cmd.CombinedOutput()
	if err != nil { log.Fatal(err) }
}

func predictGridPoints (pathToPredictProgram, pathToGridFile, pathToModelFile, pathToPreditionFile string) {
	fmt.Println("predicting to file", pathToPreditionFile)
	cmd := exec.Command("sh", "-c", 
		pathToPredictProgram+" "+pathToGridFile+" "+pathToModelFile+" "+pathToPreditionFile)
	_, err := cmd.CombinedOutput()
	if err != nil { log.Fatal(err) }
}

func transformationOfOrderN (x []float64, n int) []float64 {
	x2 := []float64{}
	for i := 0; i <= n; i++ {
		for j := 0; j <= i; j++ {
			x2 = append(x2, math.Pow(x[1], float64(j))*math.Pow(x[2], float64(i-j)))
		}
	}
	return x2
}

func tryLinearModelAndDrawResults (pathToDataFile, pathToModelAreaImageFile string, scale float64, 
numberOfThreadsToUse int) {
	
	labels, xs := readPoints(pathToDataFile, 0)

	semaphore := make(chan struct{}, numberOfThreadsToUse)
	var wg sync.WaitGroup

	for k := 2; k < 11; k++ {
		wg.Add(1)
		semaphore <- struct{}{}
		go func (k int) {
			defer wg.Done()
			defer func () { <-semaphore }()

			points := createPointsFromData(labels, xs)
			transformationFunction := func (x []float64) []float64 { return transformationOfOrderN(x, k) }
			for i := range points { points[i].transform(transformationFunction) }
			weights, iterations, inError := pla(points, nil, 100000)
			fmt.Printf("transformation order: %v, iterations: %v, inError: %v\n", k, iterations, inError)

			img := buildImageBigEnoughToFitPoints(xs, scale)
			bounds := img.Bounds()
			for j := bounds.Min.Y; j < bounds.Max.Y; j++ {
				for i := bounds.Min.X; i < bounds.Max.X; i++ {
					gridPointParameters := transformationOfOrderN([]float64{1.0, float64(i)/scale, float64(j)/scale}, k)
					label := 0
					if dotProduct(weights, gridPointParameters) > 0 { label = 1 } else { label = -1 }
					addColorAt(i, j,img, color.RGBA{(255/2)*uint8(1+label), (255/2)*uint8(1-label), 0, 255})
				}
			}

			drawPointsInImage(labels, xs, img, scale)

			pathToModelAreaImageFile2 := pathToModelAreaImageFile[:len(pathToModelAreaImageFile)-4]+
				"Order"+fmt.Sprintf("%02d", k)+pathToModelAreaImageFile[len(pathToModelAreaImageFile)-4:]
			saveImage(img, pathToModelAreaImageFile2)

		}(k)

	}
	wg.Wait()
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
	pathToPreditionFile := path+predictionFileName
	pathToModelAreaImageFile := "bananaModelArea.png"

	convertDataToLibsvmFormat(pathToOriginalFormatFile, pathToDataFile)

	scale := 100.0
	img := drawPoints(pathToDataFile, pathToDistributionImageFile, scale)
	saveGridToFile(pathToGridFile, img, scale)
	selectedCombinations := findBestModel(pathToTrainProgram, pathToDataFile, numberOfThreadsToUse)

	var wg sync.WaitGroup
	for _, combination := range selectedCombinations {
		pathToModelFile2 := pathToModelFile[:len(pathToModelFile)-4]+
			combination.getKernelName()+pathToModelFile[len(pathToModelFile)-4:]
		pathToPreditionFile2 := pathToPreditionFile[:len(pathToPreditionFile)-4]+
			combination.getKernelName()+pathToPreditionFile[len(pathToPreditionFile)-4:]
		pathToModelAreaImageFile2 := pathToModelAreaImageFile[:len(pathToModelAreaImageFile)-4]+
			combination.getKernelName()+pathToModelAreaImageFile[len(pathToModelAreaImageFile)-4:]
		wg.Add(1)
		go func (pathToModelFile, pathToPreditionFile, pathToModelAreaImageFile string, combination Combination) {
			defer wg.Done()
			train(pathToTrainProgram, pathToDataFile, pathToModelFile, combination)
			predictGridPoints(pathToPredictProgram, pathToGridFile, pathToModelFile, pathToPreditionFile)
			img2 := drawModelAreaWithPoints(pathToDataFile, pathToPreditionFile, pathToModelAreaImageFile, scale)
			drawSupportVectorsOnTopOfPoints(img2, pathToModelFile2, pathToModelAreaImageFile, scale)
		}(pathToModelFile2, pathToPreditionFile2, pathToModelAreaImageFile2, combination)
	}
	wg.Wait()

	tryLinearModelAndDrawResults(pathToDataFile, pathToModelAreaImageFile, scale, numberOfThreadsToUse)

}
