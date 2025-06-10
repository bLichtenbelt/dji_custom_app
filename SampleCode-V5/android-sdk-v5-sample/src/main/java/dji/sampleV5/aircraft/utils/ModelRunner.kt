package dji.sampleV5.aircraft.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import com.microsoft.onnxruntime.OnnxTensor
import com.microsoft.onnxruntime.OrtSession
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer

object ModelRunner {
    private var session: OrtSession? = null

    /**
     * Initialize the ONNX model using a file from the assets folder.
     */
    fun init(context: Context, assetPath: String) {
        session = OnnxModelLoader.loadModel(context, assetPath)
    }

    /**
     * Run the loaded model on a YUV frame and return the raw output.
     */
    fun run(frameData: ByteArray, width: Int, height: Int): Array<FloatArray>? {
        val sess = session ?: return null
        val bitmap = yuvToBitmap(frameData, width, height)
        val inputTensor = preprocess(bitmap)
        val output = sess.run(listOf(inputTensor))
        val result = output[0].value as Array<FloatArray>
        inputTensor.close()
        output.close()
        return result
    }

    private fun yuvToBitmap(data: ByteArray, width: Int, height: Int): Bitmap {
        val yuvImage = YuvImage(data, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
        val jpegData = out.toByteArray()
        return BitmapFactory.decodeByteArray(jpegData, 0, jpegData.size)
    }

    private fun preprocess(bitmap: Bitmap): OnnxTensor {
        val w = bitmap.width
        val h = bitmap.height
        val floatBuffer = FloatBuffer.allocate(w * h * 3)
        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val idx = y * w + x
                val px = pixels[idx]
                floatBuffer.put(((px shr 16) and 0xFF) / 255.0f)
                floatBuffer.put(((px shr 8) and 0xFF) / 255.0f)
                floatBuffer.put((px and 0xFF) / 255.0f)
            }
        }
        floatBuffer.rewind()
        return OnnxTensor.createTensor(session!!.environment, floatBuffer, longArrayOf(1, 3, h.toLong(), w.toLong()))
    }
}
