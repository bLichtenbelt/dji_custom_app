package dji.sampleV5.aircraft.utils

import android.content.Context
import com.microsoft.onnxruntime.OrtEnvironment
import com.microsoft.onnxruntime.OrtSession

object OnnxModelLoader {
    private var env: OrtEnvironment? = null

    /**
     * Load an ONNX model from the assets folder and create an OrtSession.
     *
     * @param context The Android context used to access assets.
     * @param assetPath Path to the ONNX model file inside the assets folder.
     * @return OrtSession ready for inference.
     */
    fun loadModel(context: Context, assetPath: String): OrtSession {
        if (env == null) {
            env = OrtEnvironment.getEnvironment()
        }
        val modelBytes = context.assets.open(assetPath).use { it.readBytes() }
        return env!!.createSession(modelBytes)
    }

    /**
     * Load an ONNX model from a file path on external storage.
     *
     * @param filePath Absolute path to the ONNX model file.
     * @return OrtSession ready for inference.
     */
    fun loadModelFromFile(filePath: String): OrtSession {
        if (env == null) {
            env = OrtEnvironment.getEnvironment()
        }
        val modelBytes = java.io.File(filePath).readBytes()
        return env!!.createSession(modelBytes)
    }
}
