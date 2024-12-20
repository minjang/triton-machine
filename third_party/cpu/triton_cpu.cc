#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#if defined(__x86_64__) || defined(__i386__)
#include <asm/prctl.h>
#endif
#include <sys/syscall.h>
#include <unistd.h>

namespace py = pybind11;

void init_triton_cpu_passes_ttmir(py::module &&m) {
  using namespace mlir::triton;
  // NOTE: We still use TritonGPUToLLVM pass here as a PoC. We will eventually
  // have TritonMachineLLVM and TritonCPUToLLVM.
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonGPUToLLVMPassForCPU());
  });
}

void init_triton_cpu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_cpu_passes_ttmir(passes.def_submodule("ttmir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("find_kernel_names", [](mlir::ModuleOp &mod) {
    std::vector<std::string> res;
    mod.walk([&](mlir::FunctionOpInterface funcOp) {
      // Kernel functions are public and have a body.
      if (!funcOp.getFunctionBody().empty() &&
          funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public)
        res.push_back(funcOp.getName().str());
    });
    return res;
  });
}
