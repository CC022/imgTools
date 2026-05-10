//
//  DebugLog.swift
//  panoDev
//
//  Appends timestamped log lines to /tmp/panoDev_debug.log.
//  NSApplication swallows print() so we write to disk instead.
//

import Foundation

private let logPath = "/tmp/panoDev_debug.log"
private let logHandle: FileHandle? = {
    FileManager.default.createFile(atPath: logPath, contents: nil)
    return FileHandle(forWritingAtPath: logPath)
}()
private let logQ = DispatchQueue(label: "panoDev.debuglog")

func dbg(_ message: String,
         file: String = #fileID,
         line: Int = #line) {
    let ts = String(format: "%.3f", Date().timeIntervalSinceReferenceDate)
    let text = "[\(ts)] \(file):\(line)  \(message)\n"
    logQ.async {
        logHandle?.write(text.data(using: .utf8)!)
    }
    // Also attempt stdout (works in Xcode console)
    print(text, terminator: "")
}
