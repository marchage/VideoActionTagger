// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "VideoActionTagger",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(name: "video-action-tagger", targets: ["VideoActionTagger"])
    ],
    dependencies: [
        // No external dependencies to keep it simple
    ],
    targets: [
        .executableTarget(
            name: "VideoActionTagger",
            path: "Sources"
        )
    ]
)
