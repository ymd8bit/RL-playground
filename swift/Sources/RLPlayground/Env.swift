import Python
import TensorFlow


public protocol EnvProtocol {
    func play()
}


public class EnvClass {
    public var env: PythonObject
    public var name: String
    public var numEpisodes: Int 
    public var enableRender: Bool
    public var useSaved: String?
    public var numStates: Int
    public var numActions: Int

    init(_ name: String,
        _ env: PythonObject,
        _ numEpisodes: Int,
        _ numStates: Int,
        _ numActions: Int,
        enableRender: Bool = true,
        useSaved: Optional<String> = nil) 
    {
        self.name = name
        self.env = env
        self.numEpisodes = numEpisodes
        self.numStates = numStates
        self.numActions = numActions
        self.enableRender = enableRender
        self.useSaved = useSaved
    }
}

public typealias Env = EnvClass & EnvProtocol