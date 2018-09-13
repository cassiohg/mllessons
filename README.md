1- download LIBSVM from: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
2- install using their instructions.

3- Install go: https://golang.org/

4- Download this repository using go.
  go install https://github.com/cassiohg/mllessons

5- copy libsvm programs to mllessons folder
  cp path/to/svm-train path/to/mllessons
  cp path/to/svm-predict path/to/mllessons

then
  cd path/to/mllessons
6- either
  go run *.go
6- or
  go install
  ./mlclass
