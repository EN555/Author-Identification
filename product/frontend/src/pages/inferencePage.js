import { Button } from 'react-bootstrap';
import {useEffect, useState} from 'react'
import InferForm from '../components/inferForm'
import { getInferences } from '../services/api';
import { Route } from "react-router-dom";

export default function InferencePage() {
  const [inferences,setInferences] = useState([]);
  const [loading,setLoading] = useState(false);
  const fetch_inferences = async()=>{
    setLoading(true);
    try{
      const new_infrences = await getInferences();
      setInferences(new_infrences["data"]);
    }catch{}
    setLoading(false);
  }
  useEffect(()=>{
    fetch_inferences();
  },[]);

  return (
    <div>
        {inferences.length===0 ? "no inferences to show..." : inferences.map((curr,index)=>(
          <p key={index}>curr</p>
        ))}
        <Route render={({ history}) => (
          <Button onClick={() => { history.push('/retrain') }}>Retrain Page</Button>
          )} />
        <InferForm />
    </div>
  )
}
