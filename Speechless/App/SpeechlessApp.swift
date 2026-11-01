import SwiftUI

@main
struct SpeechlessApp: App {
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
