# tools/tinympc — TinyMPC（third-party vendor）

Vendored, unmodified third-party source of the TinyMPC MPC solver.

Source: https://github.com/TongjiSuperPower/sp_vision_25
Path: `tasks/auto_aim/planner/tinympc/`

Files vendored (kept unmodified):

- `admm.hpp` / `admm.cpp`          ADMM solver core
- `tiny_api.hpp` / `tiny_api.cpp`  C API（setup / solve / bounds / settings）
- `types.hpp`                      `TinySolver` / `TinyWorkspace` / `TinyCache` / `TinySettings` 等结构体
- `tiny_api_constants.hpp`         默认 settings 常量
- `rho_benchmark.hpp` / `rho_benchmark.cpp` adaptive-rho 辅助（`admm.cpp` 无条件依赖其类型/符号，需一并链接）

Excluded (not needed at runtime):

- `codegen.cpp`     嵌入式代码生成（仅 `USING_CODEGEN` 路径需要）
- `CMakeLists.txt`  upstream 构建脚本

License / attribution: 见上游仓库（TinyMPC 为 MIT 许可），保留归属说明。

注意事项：

- `types.hpp` 使用全局 `using namespace Eigen;`；结构体为普通 C 结构体。
- `TinySolver` 持有 4 个 `new` 分配的子结构体（solution/cache/settings/work），
  upstream 未提供 cleanup API；释放由 Robocore 自研封装 `tools/tiny_mpc_2d.{hpp,cpp}` 负责。
- **不要修改这些文件**，除非确有必要（并在此 README 记录原因）。
