import http from "./httpWrapper";
import config from "../config.json";

const headers = {
  headers: {
    'accept': 'application/json',
    'Content-Type': 'application/json',
  }
}

export function infer(data) {
  return http.post(`${config.apiUrl}/infer`,data,headers);
}

export function getInferences(){
  return http.get(`${config.apiUrl}/inferences`);
}

export function getInferencesInputExamples(){
  return http.get(`${config.apiUrl}/examples/inference`);
}

export function getModels(){
  return http.get(`${config.apiUrl}/models`);
}

export function updateModels(model_id){
  return http.put(`${config.apiUrl}/model?model_id=${model_id}`);
}

export function retrain(data){
  return http.post(`${config.apiUrl}/retrain`,data,headers);
}