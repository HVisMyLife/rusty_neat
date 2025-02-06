use crate::{Genre, NN};
use simplesvg as svg;
use std::fs;

pub fn svg_nn(nn: &NN, save: bool, id: usize) -> String {
    let mut objs: Vec<svg::Fig> = vec![];
    let mut positions: Vec<(f32, f32)> = vec![(0_f32, 0_f32); nn.nodes.len()];

    let len_max = nn.layer_order.iter().map(|l| l.len() ).max().unwrap();
    nn.layer_order.iter().enumerate().for_each(|(x, l)|{
        let offset = (len_max as f32 - l.len() as f32) / 2.0; // vertical centering

        l.iter().enumerate().for_each(|(y, p)|{
            positions[*p] = (
                (x as f32 + 1.0) * 64.0, 
                (y as f32 + 1.0 + offset ) * 64.0
            );

            let mut cir = svg::Fig::Circle(positions[*p].0, positions[*p].1, 12.0);
            let mut att = svg::Attr::default();
            match nn.nodes[*p].genre {
                Genre::Input => att = svg::Attr::fill(att, svg::ColorAttr::Color(0, 200, 200 * (x == 0 && y == 0) as u8)), // bias node 
                Genre::Hidden => att = svg::Attr::fill(att, svg::ColorAttr::Color(100, 100, 100)),
                Genre::Output => att = svg::Attr::fill(att, svg::ColorAttr::Color(127, 0, 0)),
            }
            cir = cir.styled(att);
            objs.push(cir);

            let txt = (*p).to_string();
            let mut nr = svg::Fig::Text(positions[*p].0 - 4. * txt.len() as f32, positions[*p].1 + 4., txt);
            let att = svg::Attr::default();
            nr = nr.styled(att);
            objs.push(nr);
        });
    });

    nn.connections.iter().filter(|e| e.active).for_each(|c| {
        let mut lin = svg::Fig::Line(
            positions[c.from].0,
            positions[c.from].1,
            positions[c.to].0,
            positions[c.to].1,
        );
        let mut att = svg::Attr::default();
        att = svg::Attr::stroke(att, svg::ColorAttr::Color(
            !c.recurrent as u8 * 255,// (c.weight > 0.0) as u8 * 255, 
            0, 
            c.recurrent as u8 * 255));// (c.weight < 0.0) as u8 * 255));
        att = svg::Attr::stroke_width(att, (c.weight * 4.0).abs().clamp(0.15, 4.0) as f32);
        lin = lin.styled(att);
        objs.insert(0, lin); // nodes above connections
    });

    let size = (
        (nn.layer_order.len() + 1) as f32 * 64.,
        (len_max + 1) as f32 * 64.,
    );
    let mut bg = svg::Fig::Rect(0., 0., size.0, size.1);
    let mut att = svg::Attr::default();
    att = svg::Attr::fill(att, svg::Color(24, 24, 24));
    bg = bg.styled(att);
    objs.insert(0, bg);

    let out = svg::Svg{0: objs, 1: size.0 as u32, 2: size.1 as u32}.to_string();
    if save {
        let path = "nn".to_owned() + &id.to_string() + ".svg";
        fs::write(&path, out.clone()).unwrap();
    }
    out
}

