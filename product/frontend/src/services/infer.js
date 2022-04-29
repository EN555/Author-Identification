import http from "./httpWrapper";
import config from "../config.json";
const EndPoint = config.apiUrl + "/infer";

export function infer(data) {
  return http.post(EndPoint,data);
}
