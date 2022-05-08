import { useEffect,useState } from 'react'
import { getModels,updateModels } from '../services/api';
import Table from 'react-bootstrap/Table'
import { Button,Modal } from 'react-bootstrap';
import { toast } from 'react-toastify';

export default function ModelDashboardPage() {
  const [trainedModels,setTrainedModels] = useState([]);
  const [selectedModel,setSelectedModel] = useState();
  const [loading,setLoading] = useState(false);
  const [showModal,setShowModal] = useState(false);
  console.log(loading);
  
  const updateModelHandler = async()=>{
    setShowModal(false);
    setLoading(true);
    try{
      if(selectedModel) await updateModels(selectedModel.id);
    }catch(e){
      console.log(e);
      toast.error("Faild to update model")
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
        <h1 style={{marginBottom: "2rem"}}>Models</h1>
        <Modal
          size="l"
          aria-labelledby="contained-modal-title-vcenter"
          centered
          show={showModal}
          onHide={()=>setShowModal(false)}
        >
          <Modal.Header closeButton>
            <Modal.Title id="contained-modal-title-vcenter">
              Deploy Model
            </Modal.Title>
          </Modal.Header>
          <Modal.Body>
            <h4>Are You Sure you want to deploy?</h4>
            <p>
              updating this model will take some time...
            </p>
          </Modal.Body>
          <Modal.Footer>
        <Button variant='primary' onClick={updateModelHandler}>Yes</Button>
        <Button variant='secondary' onClick={()=>setShowModal(false)}>No</Button>
      </Modal.Footer>
    </Modal>
      {trainedModels.length === 0 ? "no models to show..." : (
        <>
          <Table stripped bordered hover variant="dark" size="sm">
          <thead>
            <tr>
              <th width="10%">Model Id</th>
              <th width="10%">Train_accuracy</th>
              <th width="10%">Test accuracy</th>
              <th width="10%">Train duration(seconds)</th>
              <th width="10%">Dataset Size</th>
              <th width="10%">Created At</th>
            </tr>
          </thead>
          <tbody>
            {trainedModels.map((curr_model,index) => (
            <tr className={(selectedModel && curr_model.id === selectedModel.id) ? "selected-row" : ""} key={index} onClick={()=>setSelectedModel(curr_model)}>
              <td>{curr_model.id}</td>
              <td>{curr_model.train_acc}</td>
              <td>{curr_model.test_acc}</td>
              <td>{curr_model.duration}</td>
              <td>{curr_model.dataset.size}</td>
              <td>{curr_model.created_at}</td>
            </tr>
            ))}
          </tbody>
          </Table>
          <Button variant="primary" onClick={()=>setShowModal(true)} disabled={!selectedModel}>Retrain Selected</Button>
        <Button variant="secondary" style={{width: "5rem",marginLeft:"1rem"}} onClick={()=>setSelectedModel(null)} disabled={!selectedModel}>Clear</Button>
          </>
        )
        }
    </div>
  )
}
