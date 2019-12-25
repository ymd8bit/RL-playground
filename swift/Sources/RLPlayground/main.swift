func playMaze()  {
  // let maze = Maze("maze5x5", enableRender: false)
  // let maze = Maze("maze5x5", enableRender: true, useSaved: "maze5x5.npy")
  let maze = Maze("maze5x5", enableRender: false, useSaved: "maze5x5.npy")
  let (_, time) = timeMeasure(maze.play())
  print(String(format: "time: %.10f s", time))
}

func playTaxi() {
  // let taxi = Taxi("taxi", enableRender: false)
  // let taxi = Taxi("taxi", enableRender: true, useSaved: "taxi.npy")
  let taxi = Taxi("taxi", enableRender: false, useSaved: "taxi.npy")
  let (_, time) = timeMeasure(taxi.play())
  print(String(format: "time: %.10f s", time))
}

func playCartPole() {
  let cartPole = CartPole("cartPole", numEpisodes: 1, enableRender: true)
  cartPole.play()
}

// playMaze()
// playTaxi()
playCartPole()
