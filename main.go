package main

import (
  "os"
  "fmt"
  "errors"
  "bufio"
  "github.com/goml/gobrain"
  "log"
  "github.com/matryer/persist"
  "strings"
  "math/rand"
)

func loadData() ([][]float64, []string, error){
  f, err := os.Open("iris.csv")
  if err != nil {
    return nil, nil, err
  }
  defer f.Close()

  scanner := bufio.NewScanner(f)
  // skip 
  scanner.Scan()

  var resultf [][]float64
  var results []string

  for scanner.Scan() {
    var f1, f2, f3, f4 float64
    var s string
    n, err := fmt.Sscanf(scanner.Text(), "%f,%f,%f,%f,%s", &f1, &f2, &f3, &f4, &s)
    if n != 5 || err != nil {
      return nil, nil, errors.New("cannnot load data")
    }

    resultf = append(resultf, []float64{f1, f2, f3, f4})
    results = append(results, strings.Trim(s, "\""))
  }
  return resultf, results, nil
}

func shuffle(x [][]float64, y []string) {
  for i := len(x)-1; i >= 0; i -- {
    j := rand.Intn(i+1)
    x[i], x[j] = x[j], x[i]
    y[i], y[j] = y[j], y[i]
  }
}

func main() {
  X, Y, err := loadData()
  if err != nil {
    log.Fatal(err)
  }

  shuffle(X, Y)
  //fmt.Println(Y)
  //os.Exit(0)

  n := 100
  xtrain, ytrain, xtest, ytest := X[:n], Y[:n],X[:n], Y[:n]
  patterns := [][][]float64{}

  M := map[string][]float64 {
    "Setosa": []float64{1, 0, 0},
    "Versicolor": []float64{0, 1, 0},
    "Virginica": []float64{0, 0, 1},
  }

  for i, x := range xtrain {
    patterns = append(patterns, [][]float64 {
      x, M[ytrain[i]],
    })
  }
  // instantiate the Feed Forward
  ff := &gobrain.FeedForward{}
  // initialize the Neural Network;
  // the networks structure will contain:
  // 2 inputs, 2 hidden nodes and 1 output.
  ff.Init(4, 3, 3)

  err = persist.Load("model.json", &ff)
  if err != nil {
    // train the network using the XOR patterns
    // the training will run for 1000 epochs
    // the learning rate is set to 0.6 and the momentum factor to 0.4
    // use true in the last parameter to receive reports about the learning error
    ff.Train(patterns, 100000, 0.6, 0.04, true)
    persist.Save("model.json", &ff)
  }

  var a int
  for i, x := range xtest {
    result := ff.Update(x)
    var mf float64
    var mi int
    for i, v := range result {
      if mf < v {
        mf = v
        mi = i
      }
    }
    want := ytest[i]
    got := []string{"Setosa", "Versicolor", "Virginica"}[mi]
    if want == got {
      a++
    }
    //fmt.Println(result)
    //fmt.Println([]string{"Setosa", "Versicolor", "Virginica"}[mi])
  }
  fmt.Printf("%.02f%%\n", float64(a)/float64(len(xtest))*100)
}

