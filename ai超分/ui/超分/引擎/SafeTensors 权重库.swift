import Foundation

/// safetensors 容器解析。
///
/// 文件布局：`[8 字节小端 header 长度][header JSON][连续张量数据段]`。
/// header 是 `{ 张量名: { dtype, shape, data_offsets } }`，data_offsets 相对数据段起点。
/// 这里不做拷贝式解码，只在需要时把目标张量转成连续 Float 数组，供 Metal buffer 上传。
nonisolated struct SafeTensors权重库 {

    enum 存储类型: String {
        case f32 = "F32"
        case f16 = "F16"
        case bf16 = "BF16"
        case i64 = "I64"
        case i32 = "I32"
        case u8 = "U8"
        case bool = "BOOL"
    }

    struct 张量头 {
        let 名称: String
        let 类型: 存储类型
        let 形状: [Int]
        /// 相对数据段起点的字节偏移
        let 数据起点: Int
        let 字节数: Int

        var 元素数: Int { 形状.reduce(1, *) }
    }

    /// 只读内存映射，避免 9.4 GB 权重被复制进堆
    private let 文件数据: Data
    private let 数据段起点: Int
    private let 头部表: [String: 张量头]
    /// 保留顺序，便于排查权重是否缺失
    let 张量名: [String]

    var 元数据: [String: String] { _元数据 }
    private let _元数据: [String: String]

    init(路径: String) throws {
        let url = URL(fileURLWithPath: 路径)
        let 数据 = try Data(contentsOf: url, options: [.mappedIfSafe])
        guard 数据.count >= 8 else { throw 错误.头部损坏("文件长度不足 8 字节") }

        let 头长度 = 数据.withUnsafeBytes { 缓冲区 -> UInt64 in
            缓冲区.loadUnaligned(fromByteOffset: 0, as: UInt64.self).littleEndian
        }
        guard 头长度 > 0, Int(头长度) <= 数据.count - 8 else {
            throw 错误.头部损坏("header 长度 \(头长度) 越界")
        }

        let 头切片 = 数据.subdata(in: 8..<(8 + Int(头长度)))
        guard let 原始 = try JSONSerialization.jsonObject(with: 头切片) as? [String: Any] else {
            throw 错误.头部损坏("header 不是 JSON 对象")
        }

        var 表: [String: 张量头] = [:]
        var 顺序: [String] = []
        var 元: [String: String] = [:]

        for (键, 值) in 原始 {
            if 键 == "__metadata__" {
                元 = (值 as? [String: String]) ?? [:]
                continue
            }
            guard let 描述 = 值 as? [String: Any],
                  let 类型串 = 描述["dtype"] as? String,
                  let 类型 = 存储类型(rawValue: 类型串),
                  let 形状 = 描述["shape"] as? [Int],
                  let 偏移 = 描述["data_offsets"] as? [Int], 偏移.count == 2
            else { continue }

            let 起点 = 偏移[0], 终点 = 偏移[1]
            表[键] = 张量头(名称: 键, 类型: 类型, 形状: 形状,
                             数据起点: 起点, 字节数: 终点 - 起点)
            顺序.append(键)
        }

        文件数据 = 数据
        数据段起点 = 8 + Int(头长度)
        头部表 = 表
        张量名 = 顺序.sorted()
        _元数据 = 元
    }

    enum 错误: LocalizedError {
        case 头部损坏(String)
        case 张量不存在(String)
        case 尺寸不符(String)
        case 类型暂不支持(String)

        var errorDescription: String? {
            switch self {
            case .头部损坏(let 说明): "权重文件头部损坏：\(说明)"
            case .张量不存在(let 名): "权重文件缺少张量：\(名)"
            case .尺寸不符(let 名): "张量字节数与形状不匹配：\(名)"
            case .类型暂不支持(let 说明): "暂不支持的权重类型：\(说明)"
            }
        }
    }

    func 头(_ 名: String) throws -> 张量头 {
        guard let 描述 = 头部表[名] else { throw 错误.张量不存在(名) }
        return 描述
    }

    func 是否包含(_ 名: String) -> Bool { 头部表[名] != nil }

    /// 元数据里的 JSON 配置（RelayVSR 把模型超参写在这里）
    func 配置<T: Decodable>(_ 类型: T.Type, 键: String = "config") throws -> T {
        guard let 文本 = _元数据[键], let 原始 = 文本.data(using: .utf8) else {
            throw 错误.头部损坏("缺少元数据字段 \(键)")
        }
        return try JSONDecoder().decode(T.self, from: 原始)
    }

    /// 把任意 dtype 的张量解码为连续 Float32（bf16 走左移补位，不做舍入）
    func Float32张量(_ 名: String) throws -> (形状: [Int], 数值: [Float]) {
        let 描述 = try 头(名)
        let 有效字节 = 数据段起点 + 描述.数据起点
        guard 有效字节 + 描述.字节数 <= 文件数据.count else {
            throw 错误.尺寸不符(名)
        }
        let 元素数 = 描述.元素数
        var 输出 = [Float](repeating: 0, count: 元素数)

        文件数据.withUnsafeBytes { 缓冲区 in
            let 基址 = 缓冲区.baseAddress!.advanced(by: 有效字节)
            switch 描述.类型 {
            case .f32:
                let 指针 = 基址.assumingMemoryBound(to: UInt32.self)
                for i in 0..<元素数 {
                    输出[i] = Float(bitPattern: 指针[i].littleEndian)
                }
            case .f16:
                let 指针 = 基址.assumingMemoryBound(to: UInt16.self)
                for i in 0..<元素数 {
                    输出[i] = Float(Float16(bitPattern: 指针[i].littleEndian))
                }
            case .bf16:
                let 指针 = 基址.assumingMemoryBound(to: UInt16.self)
                for i in 0..<元素数 {
                    let 位 = UInt32(指针[i].littleEndian) << 16
                    输出[i] = Float(bitPattern: 位)
                }
            case .i32:
                let 指针 = 基址.assumingMemoryBound(to: Int32.self)
                for i in 0..<元素数 { 输出[i] = Float(指针[i].littleEndian) }
            default:
                break
            }
        }
        return (描述.形状, 输出)
    }
}
