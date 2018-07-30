package main

import (
	"fmt"
	// "time"
	"math/rand"
)

type Point struct {
	x, y	float32
	class 	int
}

type TargetFunction struct {
	p1, p2 		Point
	slope, b	float32
}

func (f *TargetFunction) calculateFunctionFromPoints () {
	f.slope = (f.p2.y - f.p1.y) / (f.p2.x - f.p1.x)
	f.b = f.p1.y - f.slope*f.p1.x
}

func (f TargetFunction) classifyPoint (p *Point) int {
	if p.y > f.slope*p.x + f.b { p.class = 1 } else { p.class = -1 }
	return p.class
}

func main() {
	fmt.Println("Assignments for computer intelligence course on UFRJ 2018\n")

	rand.Seed(0)

	f := TargetFunction{
		p1: Point{x: rand.Float32(), y: rand.Float32()}, 
		p2: Point{x: rand.Float32(), y: rand.Float32()}}
	f.calculateFunctionFromPoints()

	amountOfPoints := 100
	points := make([]Point, amountOfPoints)
	for i := range points {
		points[i].x, points[i].y = rand.Float32(), rand.Float32()
		f.classifyPoint(&points[i])
	}

}

