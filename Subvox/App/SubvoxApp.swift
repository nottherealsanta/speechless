import SwiftUI

@main
struct SubvoxApp: App {
    @State private var studio = Studio()

    var body: some Scene {
        WindowGroup {
            StudioView()
                .environment(studio)
                .frame(minWidth: 980, minHeight: 720)
        }
        .windowResizability(.contentMinSize)
    }
}
