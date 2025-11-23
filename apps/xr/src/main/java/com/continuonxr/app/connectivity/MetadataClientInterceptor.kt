package com.continuonxr.app.connectivity

import io.grpc.*

/**
 * Simple interceptor to attach static metadata (e.g., auth).
 */
class MetadataClientInterceptor(private val headers: Metadata) : ClientInterceptor {
    override fun <ReqT : Any?, RespT : Any?> interceptCall(
        method: MethodDescriptor<ReqT, RespT>,
        callOptions: CallOptions,
        next: Channel
    ): ClientCall<ReqT, RespT> {
        val call = next.newCall(method, callOptions)
        return object : ForwardingClientCall.SimpleForwardingClientCall<ReqT, RespT>(call) {
            override fun start(responseListener: Listener<RespT>, headers: Metadata) {
                headers.merge(this@MetadataClientInterceptor.headers)
                super.start(responseListener, headers)
            }
        }
    }
}

