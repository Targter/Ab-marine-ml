

## API Endpoints

### 1. **Home**

   - **Endpoint:** `/`
   - **Method:** `GET`
   - **Description:** Returns a welcome message.
   - **Response:**

     ```json
     {
       "message": "Send a POST request to /get_image with plain text as the body to get an image."
     }
     ```

### 2. **Get Image**

   - **Endpoint:** `/get_image`
   - **Method:** `POST`
   - **Description:** Accepts plain text input and returns the most similar image.

   #### Request

   - **Headers:**

     ```
     Content-Type: application/json
     ```

   - **Body:**

     Raw text (e.g., `"A scenic mountain view"`).

   #### Response

   - **Success (200):**

     Returns the image file.

   - **Error (400):**

     ```json
     {
       "error": "Text input is required"
     }
     ```

   - **Error (404):**

     ```json
     {
       "detail": "Image not found"
     }
     ```

---

## Integration Guide for Frontend Developers

### Making API Requests

#### Using `fetch` (JavaScript):

```javascript
const getImage = async (text) => {
  const response = await fetch("http://127.0.0.1:5000/get_image", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(text),
  });

  if (response.ok) {
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);
    document.getElementById("image-display").src = imageUrl;
  } else {
    const error = await response.json();
    console.error("Error:", error);
  }
};

// Example usage:
getImage("A scenic mountain view");
```

#### Using `axios` (JavaScript):

```javascript
import axios from "axios";

const getImage = async (text) => {
  try {
    const response = await axios.post("http://127.0.0.1:5000/get_image", text, {
      headers: {
        "Content-Type": "application/json",
      },
      responseType: "blob", // Important to handle the image file
    });

    const imageUrl = URL.createObjectURL(response.data);
    document.getElementById("image-display").src = imageUrl;
  } catch (error) {
    console.error("Error:", error.response.data);
  }
};

// Example usage:
getImage("A scenic mountain view");
```

### Frontend HTML Example

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Image Retrieval</title>
</head>
<body>
  <h1>Image Retrieval</h1>
  <input type="text" id="text-input" placeholder="Enter a description" />
  <button onclick="submitText()">Get Image</button>
  <div>
    <img id="image-display" alt="Retrieved Image" style="max-width: 100%; margin-top: 20px;" />
  </div>

  <script>
    async function submitText() {
      const text = document.getElementById("text-input").value;

      const response = await fetch("http://127.0.0.1:5000/get_image", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(text),
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        document.getElementById("image-display").src = imageUrl;
      } else {
        const error = await response.json();
        console.error("Error:", error);
      }
    }
  </script>
</body>
</html>
```

---

## Notes

1. Ensure the `images` directory contains relevant images.
2. The API is designed for local use. To make it publicly accessible, deploy it to a server and update the frontend URL accordingly.
3. Test thoroughly in both development and production environments.

---

## Troubleshooting

1. **No images returned:**
   - Check if the `images` directory exists and contains valid image files.
   - Ensure the text input matches the context of available images.

2. **Module not found:**
   - Ensure all required packages are installed using `pip install -r requirements.txt`.

3. **CORS issues:**
   - Use a tool like [Postman](https://www.postman.com/) for testing.
   - Verify frontend and backend are communicating over the correct URLs.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
