import { useEffect,useState } from 'react'
import { getModels, retrain,updateModels } from '../services/api';

export default function RetrainPage() {
  const [trainedModels,setTrainedModels] = useState([]);
  const [selectedModel,setSelectedModel] = useState();
  const [loading,setLoading] = useState(false);
  const [retrainning,setRetrainning] = useState(false);
  const retrainHandler = async()=>{
    setRetrainning(true);
    try{
      await retrain();
    }catch(e){
      console.log(e);
    }
    setRetrainning(false);
  }
  const updateModelHandler = async()=>{
    setLoading(true);
    try{
      if(selectedModel) await updateModels(selectedModel.model_id);
    }catch(e){
      console.log(e);
    }
    setLoading(false);
  }
  const fetch_models = async()=>{
    setLoading(true);
    try{
      const models = await getModels();
      setTrainedModels(models["data"]);
    }catch{}
    setLoading(false);
  }
  useEffect(()=>{
    fetch_models();
  },[]);
  return (
    <div>
        <h1>Models</h1>
        {trainedModels.length === 0 ? "no models to show..." : trainedModels.map((curr_model,i)=>(
          <p key={i}>curr_model</p>
        ))}
    </div>
  )
}
