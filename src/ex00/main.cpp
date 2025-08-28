/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.cpp                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: joaoalme <joaoalme@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2023/08/22 15:01:18 by joaoalme          #+#    #+#             */
/*   Updated: 2023/08/22 15:01:19 by joaoalme         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <iostream>
#include "Zombie.hpp"

int     main()
{
    /* Interesting to note the scope of the instances. Jose will
       be the last destroyed bcs his scope is the main brackets
    */
    Zombie jombie("Jose");
    jombie.announce();
    /* Pedro will be created, and then destroyed when the function scope finishes */
    randomChump("Pedro");
    /* heap_zombie will be destroyed when the delete operator is used */
    Zombie      *zmb_ptr = newZombie("heap_zombie");
    // Zombie      *zmb_ptr1 = newZombie("Another heap_zombie");
    zmb_ptr->announce();
    // zmb_ptr1->announce();

    delete  zmb_ptr;

    return (0);
}

