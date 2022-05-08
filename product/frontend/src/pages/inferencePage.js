import { Button,Pagination,Table } from 'react-bootstrap';
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
        {inferences.length===0 ? "no inferences to show..." : (
          <>
          <Table stripped bordered hover variant="dark" size="sm">
          <thead>
              <tr>
                <th width="30%">Input</th>
                <th width="10%">Output</th>
                <th width="10%">Duration(Miliseconds)</th>
              </tr>
            </thead>
            <tbody>
              {inferences.map((curr_inference,index) => (
              <tr key={index}>
                <td>{curr_inference.text}</td>
                <td>{curr_inference.author_name}</td>
                <td>{curr_inference.took_time}</td>
              </tr>
              ))}
            </tbody>
          </Table>
            {/* <Pagination>
              <Pagination.First />
              <Pagination.Prev />
              <Pagination.Item>{1}</Pagination.Item>
              <Pagination.Ellipsis />
              <Pagination.Next />
              <Pagination.Last />
            </Pagination> */}
          </>
          )}
        <Route render={({ history}) => (
          <Button onClick={() => { history.push('/retrain') }}>Retrain</Button>
          )} />
        <Route render={({ history}) => (
          <Button style={{marginLeft: "1rem"}} onClick={() => { history.push('/models') }}>Model Dashboard</Button>
          )} />
        <InferForm />
    </div>
  )
}
