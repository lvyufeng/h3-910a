# Invoking FlashAttnEx: what works, what blocks

Attempted to run the registered custom op through the single-op ACL API
(`aclopCompileAndExecuteV2`) instead of building a GE graph by hand. This is
the shortest path from "op is installed" to "op produces numbers".

## The harness

`t3.cpp` loads the op by name, passes `q/k/v` (fp16 ND) and `o` (fp16 ND) plus
`l` (fp32 ND), with `ACL_ENGINE_AICORE` and `ACL_COMPILE_UNREGISTERED` plus an
explicit `opPath`, and links `libacl_op_compiler`, `libacl_op_executor`,
`libacl_mdl`, `libacl_rt`.

## Progress made, in order

Each of these was a real blocker that is now cleared:

1. **GE would not initialise at all**: `GEInitializeV2 failed`, rooted in
   `Failed to initialize TeFusion` -> `No module named 'numpy'`. GE embeds and
   imports the **system** Python (3.10.12), not the project venv (3.12), and the
   system interpreter had no scientific stack.
   Fix: `python3.10 -m pip install numpy decorator attrs psutil sympy cffi pyyaml protobuf scipy`.
2. **Then `No module named 'decorator'`** -- same cause, next missing dep.
3. **Then a hard core dump** with `TBE Subprocess[task_distribute] ... main
   process disappeared`; `TBE_PARALLEL_COMPILER=1` avoids the parallel-compile
   crash (the same workaround this project already needed for vllm-omni).
4. **Then `para_check` rejected the descriptors**: `the input[q] does not
   contain the item[ori_format]`. The TBE `check_op_params` decorator requires
   `shape`, `format`, `ori_shape`, `ori_format`, `dtype` (plus `range` when
   dynamic). The impl now normalises them.
5. **`Argument opPath must not be null`** with `ACL_COMPILE_UNREGISTERED` --
   that flag needs the path to the op implementation. Supplied
   `.../customize_impl`.
6. **`ACL_COMPILE_SYS` was wrong** for this op; it segfaulted. The op needs the
   unregistered path, i.e. GE compiles it from the TBE `.py` at run time.

Independent confirmation that the impl itself is sound: called directly under
python3.10 against the installed op path,

```
python3.10 direct_call.py   ->   BUILD OK <Tik object>
```

the TIK build completes for `Ascend910A`. So the kernel builder works when it
is driven directly.

## What still blocks

The GE pre-compile stage now fails with:

```
RuntimeError: Unsupported core type: Default
```

This fires in GE's own pre-compile, **before** the op implementation runs, so it
is not my impl raising it. The graph node is being created with core type
`Default` because this hand-rolled `aclop` path never tells GE which core to
use; the installed `aic-ascend910-ops-info.json` entry carries no `coreType`
either.

## Assessment

The chain is now long but each remaining link is identified:

* op packaged and installed for `ascend910` -- done, verified in the ops-info db
* impl builds the TIK kernel for `Ascend910A` -- done, verified by direct call
* GE needs a core type for the op before it will compile it -- **blocking**

Ways forward, cheapest first:

1. Find where GE takes core type for a custom op: possibly a `coreType` field in
   the ops-info the package does not emit, or a `ge.` option such as
   `ge.exec.coreType`, or `aclopSetModelDir` plus a compiled model rather than
   compile-on-demand. Worth one focused look at the ops-info schema of a
   built-in AiCore op and comparing.
2. Skip `aclop` and build the (tiny) GE graph, where the node can carry
   `coreType` explicitly. More code, but it is the documented path for custom
   ops and the core type can be set.
3. Go back to `ACL_COMPILE_SYS`: if the op's kernel were pre-compiled into a
   `.so` at install time, GE would not need to compile it and core type would
   not arise. The package currently ships no compiled kernel, only the TBE
   `.py`, which is why GE insists on compiling.

Note (3) is likely the intended design: the op project's `build.sh` has a
pre-compile step that this project may need to be told to run, which would turn
the TBE `.py` into a shipped kernel binary and take GE's compiler out of the
loop entirely. That is the most promising lead and should be checked before
anything else.

## Not yet established

No numerics, no timing. The kernel has still never executed on this chip.
