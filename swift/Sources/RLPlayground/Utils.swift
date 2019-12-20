import Foundation
import Python
import TensorFlow

public func timeMeasure <T> (_ f: @autoclosure () -> T) -> (result: T, duration: Double) {
    let start = CFAbsoluteTimeGetCurrent()
    let res = f()
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    return (res, Double(elapsed))
}

public func saveTensor <T: NumpyScalarCompatible> (_ aten: Tensor<T>, name: String) {
    let np = Python.import("numpy")
    let fm = FileManager.default
    let path = fm.currentDirectoryPath + "/" + name + ".npy"
    // FileManager.default.fileExists(atPath: path)
    np.save(path, aten.makeNumpyArray())
}

public func loadTensor<T: NumpyScalarCompatible>(path: String) -> Tensor<T> {
    let np = Python.import("numpy")
    let aten = Tensor<T>(numpy: np.load(path))!
    return aten
}