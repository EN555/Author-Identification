import axios from "axios";
import { toast } from "react-toastify";

// axios.defaults.headers.common["x-auth-token"] = localStorage.getItem("token");

axios.interceptors.response.use(null, (error) => {
  const expectError =
    error.response &&
    error.response.status >= 400 &&
    error.response.status < 500;
  if (!expectError) {
    toast.error("An unexpexted error accurred");
    console.log(error);
  }
  return Promise.reject(error);
});

const result = { 
  get: axios.get,
  put: axios.put,
  delete: axios.delete,
  post: axios.post,
};

export default result;
