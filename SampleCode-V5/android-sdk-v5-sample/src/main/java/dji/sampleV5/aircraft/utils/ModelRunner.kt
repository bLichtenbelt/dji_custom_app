package dji.sampleV5.aircraft.utils

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import com.microsoft.onnxruntime.OnnxTensor
import com.microsoft.onnxruntime.OrtSession
import org.json.JSONObject
import android.graphics.Canvas
import android.renderscript.Allocation
import android.renderscript.Element
import android.renderscript.RenderScript
import android.renderscript.ScriptIntrinsicBlur
import java.io.ByteArrayOutputStream
import java.nio.FloatBuffer

object ModelRunner {
    private var session: OrtSession? = null

    /**
     * Initialize the ONNX model using a file from the assets folder.
     */
    fun init(context: Context, assetPath: String) {
        BlurConfig.loadFromAsset(context, "blur_config.json")
        session = OnnxModelLoader.loadModel(context, assetPath)
    }

    /**
     * Initialize the ONNX model using a file stored on external storage.
     */
    fun initFromFile(context: Context, modelPath: String) {
        BlurConfig.loadFromAsset(context, "blur_config.json")
        session = OnnxModelLoader.loadModelFromFile(modelPath)
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

    data class BoundingBox(
        val x: Float,
        val y: Float,
        val width: Float,
        val height: Float,
        val label: String,
        val score: Float
    )

    object BlurConfig {
        var labelsToBlur: Set<String> = emptySet()

        fun loadFromAsset(context: Context, assetName: String) {
            val jsonStr = context.assets.open(assetName).bufferedReader().use { it.readText() }
            val json = JSONObject(jsonStr)
            val arr = json.optJSONArray("blur_labels")
            val list = mutableSetOf<String>()
            if (arr != null) {
                for (i in 0 until arr.length()) {
                    list.add(arr.getString(i))
                }
            }
            labelsToBlur = list
        }
    }

    fun runAndBlur(context: Context, frameData: ByteArray, width: Int, height: Int): Bitmap? {
        val output = run(frameData, width, height) ?: return null
        val bitmap = yuvToBitmap(frameData, width, height)
        val boxes = parseOutput(output)
        return applyBlur(context, bitmap, boxes)
    }

    private fun parseOutput(output: Array<FloatArray>): List<BoundingBox> {
        val result = mutableListOf<BoundingBox>()
        for (row in output) {
            if (row.size >= 6) {
                val label = row[4].toInt().toString()
                result.add(
                    BoundingBox(
                        row[0],
                        row[1],
                        row[2],
                        row[3],
                        label,
                        row[5]
                    )
                )
            }
        }
        return result
    }

    private fun applyBlur(context: Context, src: Bitmap, boxes: List<BoundingBox>): Bitmap {
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(result)
        for (box in boxes) {
            if (!BlurConfig.labelsToBlur.contains(box.label)) continue
            val left = (box.x * src.width).toInt()
            val top = (box.y * src.height).toInt()
            val right = ((box.x + box.width) * src.width).toInt()
            val bottom = ((box.y + box.height) * src.height).toInt()
            val w = right - left
            val h = bottom - top
            if (w <= 0 || h <= 0) continue
            val sub = Bitmap.createBitmap(result, left, top, w, h)
            val rs = RenderScript.create(context)
            val input = Allocation.createFromBitmap(rs, sub)
            val output = Allocation.createTyped(rs, input.type)
            val script = ScriptIntrinsicBlur.create(rs, Element.U8_4(rs))
            script.setRadius(25f)
            script.setInput(input)
            script.forEach(output)
            output.copyTo(sub)
            rs.destroy()
            canvas.drawBitmap(sub, left.toFloat(), top.toFloat(), null)
        }
        return result
    }
}
