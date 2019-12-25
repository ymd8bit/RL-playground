import Python
import TensorFlow


public protocol EnvProtocol {
    func getNumActions() -> Int
    func getNumStates() -> Int
    func play()
}


public class EnvClass {
    public var env: PythonObject
    public var name: String
    public var numEpisodes: Int 
    public var enableRender: Bool
    public var useSaved: String?

    init(_ name: String,
        _ env: PythonObject,
        _ numEpisodes: Int,
        enableRender: Bool = true,
        useSaved: Optional<String> = nil) 
    {
        self.name = name
        self.env = env
        self.numEpisodes = numEpisodes
        self.enableRender = enableRender
        self.useSaved = useSaved
    }
}

public typealias Env = EnvClass & EnvProtocol