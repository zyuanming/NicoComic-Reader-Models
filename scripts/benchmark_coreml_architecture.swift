#!/usr/bin/env swift

import CoreML
import CoreVideo
import Foundation

guard CommandLine.arguments.count == 3,
      let sampleCount = Int(CommandLine.arguments[2]),
      sampleCount > 0 else {
    fatalError("usage: benchmark_coreml_architecture.swift MODEL.mlpackage SAMPLE_COUNT")
}

let package = URL(fileURLWithPath: CommandLine.arguments[1])
let compiled = try MLModel.compileModel(at: package)
defer { try? FileManager.default.removeItem(at: compiled) }

let configuration = MLModelConfiguration()
configuration.computeUnits = .cpuOnly
let model = try MLModel(contentsOf: compiled, configuration: configuration)
guard let input = model.modelDescription.inputDescriptionsByName.first,
      let constraint = input.value.imageConstraint else {
    fatalError("model must declare one image input")
}

var buffer: CVPixelBuffer?
let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
guard CVPixelBufferCreate(
    kCFAllocatorDefault,
    constraint.pixelsWide,
    constraint.pixelsHigh,
    kCVPixelFormatType_32BGRA,
    attributes,
    &buffer
) == kCVReturnSuccess, let buffer else {
    fatalError("cannot allocate input pixel buffer")
}

let provider = try MLDictionaryFeatureProvider(
    dictionary: [input.key: MLFeatureValue(pixelBuffer: buffer)]
)
var milliseconds: [Double] = []
for index in 0...sampleCount {
    let started = ContinuousClock.now
    _ = try model.prediction(from: provider)
    if index > 0 {
        let duration = started.duration(to: .now).components
        milliseconds.append(Double(duration.seconds) * 1_000 + Double(duration.attoseconds) / 1e15)
    }
}

let sorted = milliseconds.sorted()
let result: [String: Any] = [
    "inputWidth": constraint.pixelsWide,
    "inputHeight": constraint.pixelsHigh,
    "sampleCount": sampleCount,
    "medianMilliseconds": sorted[sorted.count / 2],
    "p95Milliseconds": sorted[Int(Double(sorted.count - 1) * 0.95)],
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(decoding: data, as: UTF8.self))
