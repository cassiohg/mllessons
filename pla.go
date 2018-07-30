package main

import (
	"fmt"
	// "time"
	"math/rand"
	// "math"
)

func random () float32 {
	return (rand.Float32()*2) -1
}

type Point struct {
	x 	[]float32
	y 	float32
}

type TargetFunction struct {
	a, b		float32
	x1, y1, x2, y2	float32
}
func (f *TargetFunction) initialize () {
	f.x1, f.y1, f.x2, f.y2 = random(), random(), random(), random()
	f.a = (f.y2 - f.y1) / (f.x2 - f.x1)
	f.b = f.y1 - f.a*f.x1
}
func (f TargetFunction) classifyPoint (p Point) float32 {
	if p.x[2] > f.a*p.x[1] + f.b { return 1.0 } else { return -1.0 }
}

func createPointsAndClassify (amount int, pointSize int, f *TargetFunction) []Point {
	points := make([]Point, amount)
	for i := range points {
		p := &points[i]
		p.x = make([]float32, pointSize)
		p.x[0] = 1.0	
		for j := 1; j < pointSize; j++ {
			p.x[j] = random()
		}
		p.y = f.classifyPoint(*p)
	}
	return points
}

func pla(weightVector []float32, points []Point) ([]float32, int, int) {
	totalAmountOfMissClassifications := 0
	currentAmountOfMissClassifications := 1
	iterationCounter := 0
	for currentAmountOfMissClassifications > 0 {
		currentAmountOfMissClassifications = 0
		iterationCounter++
		for i := range points {
			p := &points[i]
			sum := float32(0)
			for j := range weightVector {
				sum += weightVector[j] * p.x[j]
			}
			if (sum > 0.0) != (p.y > 0.0) {
				currentAmountOfMissClassifications++
				for j := range weightVector {
					weightVector[j] += p.y * p.x[j]
				}
			}
		}
		totalAmountOfMissClassifications += currentAmountOfMissClassifications
		// fmt.Println(iterationCounter, weightVector, currentAmountOfMissClassifications)
	}	
	return weightVector, iterationCounter-1, totalAmountOfMissClassifications
}

func main() {
	fmt.Println("Assignments for computer intelligence course on UFRJ 2018\n")

	rand.Seed(0)

	amountOfPoints := 100
	pointSize := 3

	f := TargetFunction{}
	f.initialize()
	fmt.Printf(">> target function f:\na=%f, b=%f\n%f\t%f\n%f\t%f\n---------\n", f.a, f.b, f.x1, f.y1, f.x2, f.y2)

	points := createPointsAndClassify(amountOfPoints, pointSize, &f)
	// for i, p:= range points {
	// 	fmt.Println(i, p)
	// 	// fmt.Printf("%f\t%f\t%d\n", p.x[1], p.x[2], i)
	// }

	weights, interationsCounter, missClassificationCounter := pla(make([]float32, pointSize), points)
	fmt.Println(
		weights, 
		interationsCounter, 
		missClassificationCounter, 
		float32(missClassificationCounter) / float32(interationsCounter*amountOfPoints))

	fmt.Printf(">> hypothesis function h:\n0\t%f\n%f\t0\n---------\n", -weights[0]/weights[2], -weights[0]/weights[1])

}

