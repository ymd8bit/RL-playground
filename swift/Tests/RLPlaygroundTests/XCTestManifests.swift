import XCTest

#if !canImport(ObjectiveC)
public func allTests() -> [XCTestCaseEntry] {
    return [
        testCase(RL_playgroundTests.allTests),
    ]
}
#endif
