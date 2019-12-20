import Foundation
import Python
import TensorFlow

public class Maze {

    public enum Action: Int {
        case up = 0, down, right, left
    }
    typealias State = Int
    typealias Reward = Float

    public var name: String
    public var ε: Float, η: Float, γ: Float
    public var enableRender: Bool
    public var useSaved: String?
    public var hight: Int, width: Int, n_action: Int
    public var Q: Tensor<Float>
    private var np: PythonObject, gym: PythonObject, gym_maze: PythonObject, env: PythonObject

    init(_ name: String,
        ε: Float = 0.1,
        η: Float = 0.1,
        γ: Float = 0.9,
        enableRender: Bool = true,
        useSaved: Optional<String> = nil) 
    {
        np = Python.import("numpy")
        gym = Python.import("gym")
        gym_maze = Python.import("gym_maze")
        env = gym.make("maze-sample-5x5-v0", enable_render:enableRender)

        self.name = name
        self.ε = ε
        self.η = η
        self.γ = γ
        self.enableRender = enableRender
        self.useSaved = useSaved

        hight = Array(env.observation_space.high)![0] + 1
        width = Array(env.observation_space.high)![1] + 1
        n_action = Int(env.action_space.n)!
        let shape = TensorShape([hight * width, n_action])

        if useSaved == nil {
            Q = Tensor<Float>(repeating: 0, shape: shape)
        } else { 
            let path = useSaved!
            Q = loadTensor(path: path)
            assert(Q.shape == shape)
            print("loaded a tensor from:", path)
        }
    }

    private func εGreedy(_ s: State, ε: Float) -> Action {
        if Float.random(in: 0...1) < ε {
            let a = Int.random(in: 0..<n_action)
            return Action(rawValue: a)!
        } 
        return greedy(s)
    }

    private func greedy(_ s: State) -> Action {
        let a = Int(Q[s].argmax().scalarized())
        return Action(rawValue: a)!
    }

    private func random(_ s: State) -> Action {
        let a = Int.random(in: 0...n_action-1)
        return Action(rawValue: a)!
    }

    private func updateQBySarsa(_ s: State, _ a: Action, _ r: Reward, _ s_next: State, _ a_next: Action) {
        let a = a.rawValue
        let a_next = a_next.rawValue
        let u = (1 - η) * Q[s, a]
        let v = η * (r + γ * Q[s_next, a_next])
        Q[s_next, a] = u + v
    }   


    private func step(env: PythonObject, a: Action) -> (State, Reward, Bool, PythonObject) {
        let (s, r, done, info) = env.step(a.rawValue).tuple4
        return (makeState(s), Float(r)!, Bool(done)!, info)
    }

    public func play() {
        let numEpisodes = 3000

        for ep in 0..<numEpisodes {
            var s = makeState(env.reset())
            var a_next = random(s)
            var rTotal: Reward = 0

            while true {
                let a = a_next
                let (s_next, r, done, _) = step(env: env, a: a)

                if enableRender {
                    print("s: \(s), a: \(a), r: \(r), done: \(done)")
                    env.render()
                    usleep(100000)
                }
                a_next = εGreedy(s, ε: ε)
                updateQBySarsa(s, a, r, s_next, a_next)

                s = s_next
                rTotal += r
                if(done) { break }
            }

            if ep % 100 == 0 {
                print("episode: \(ep), total_reward: \(rTotal)")
            }
        }

        saveTensor(Q, name: name)
        print("saved a tensor in: \(name).npy")
    }


    private func makeState(_ s: PythonObject) -> State {
        let (x, y) = s.tuple2
        return Int(x)! * width + Int(y)!
    }

}