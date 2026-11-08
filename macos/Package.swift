// swift-tools-version: 6.0
import PackageDescription
let package = Package(name: "Swipe", platforms: [.macOS(.v14)], products: [.executable(name: "Swipe", targets: ["SwipeApp"])], targets: [
    .target(name: "SwipeCore"),
    .executableTarget(name: "SwipeApp", dependencies: ["SwipeCore"]),
    .testTarget(name: "SwipeAppTests", dependencies: ["SwipeApp"]),
    .testTarget(name: "SwipeCoreTests", dependencies: ["SwipeCore"], resources: [.copy("capture-fixtures.json")])
], swiftLanguageModes: [.v5])
