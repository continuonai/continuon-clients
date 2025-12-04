// Platform-aware entrypoints for Gemma runtime hot-reload hooks.
//
// The implementation uses filesystem watching + POSIX signals on IO targets
// and degrades to no-ops on web.
export 'gemma_runtime_impl.dart'
    if (dart.library.html) 'gemma_runtime_stub.dart';
