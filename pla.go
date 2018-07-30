package main

import (
	"fmt"
	// "time"
	"math/rand"
)

type Point struct {
	v [3]float32
	class 	int
}

type TargetFunction struct {
	a, b		float32
	x1, y1, x2, y2	float32
}

func (f *TargetFunction) initialize () {
	f.x1, f.y1, f.x2, f.y2 = rand.Float32(), rand.Float32(), rand.Float32(), rand.Float32()
	f.a = (f.y2 - f.y1) / (f.x2 - f.x1)
	f.b = f.y1 - f.a*f.x1
}

func (f TargetFunction) classifyPoint (p Point) int {
	if p.v[2] > f.a*p.v[1] + f.b { return 1 } else { return -1 }
}

func main() {
	fmt.Println("Assignments for computer intelligence course on UFRJ 2018\n")

	rand.Seed(0)
	amountOfPoints := 100

	f := TargetFunction{}
	f.initialize()
	fmt.Println(f)

	points := make([]Point, amountOfPoints)
	for i := range points {
		points[i].v[0], points[i].v[1], points[i].v[2] = 1, rand.Float32(), rand.Float32()
		points[i].class = f.classifyPoint(points[i])
	}

	for _, p:= range points {
		fmt.Println(p)
	}

}

