import React from 'react';
import { Button,Form } from 'react-bootstrap';
import { infer } from '../services/infer';
import CircularProgress from '@mui/material/CircularProgress';


function InferForm() {
    console.log("in inferForm");
    const on_submit = async()=>{
        await infer();
    }
    return (
        <div>
            <CircularProgress />
            <Form>
                <Form.Group className="mb-3" controlId="formBasicEmail">
                    <Form.Label>Text of Author</Form.Label>
                    <Form.Control type="text" placeholder="Enter text"/>
                </Form.Group>
                <Button variant="primary" type="submit" onSubmit={on_submit}>
                    Submit
                </Button>
            </Form>
        </div>
    )
}

export default InferForm;