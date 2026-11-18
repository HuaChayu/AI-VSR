import Foundation
import Observation

enum 输出容器: String, CaseIterable, Identifiable {
    case mp4 = "MP4"
    case mov = "MOV"

    var id: String { rawValue }
}

enum 输出编码: String, CaseIterable, Identifiable {
    case h264 = "H.264"
    case hevc = "HEVC"

    var id: String { rawValue }
}

enum 输出分辨率: String, CaseIterable, Identifiable {
    case 跟随源 = "跟随源"
    case uhd = "3840 × 2160"
    case qhd = "2560 × 1440"
    case fhd = "1920 × 1080"

    var id: String { rawValue }
}

/// 输出页 Store：先只落参数模型，不接编码器。占位期间保留字段，
/// 后续接 AVAssetWriter / VideoToolbox 时不用再改 UI 层。
@Observable
final class 输出Store {
    var 容器格式: 输出容器 = .mp4
    var 编码器: 输出编码 = .hevc
    var 分辨率: 输出分辨率 = .跟随源
    var 码率: Double = 24
    var 文件名: String = "ai超分_输出"
    var 输出目录: String = ""
    var onNote: ((String) -> Void)?

    func 试导出() {
        onNote?("输出通道尚未实现：参数已记录，编码器接入后即可导出")
    }
}
